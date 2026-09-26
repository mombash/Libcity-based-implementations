import torch
import torch.nn as nn
import torch.nn.functional as F
from logging import getLogger
from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel

from mamba_ssm import Mamba as MambaSSM


class SimpleMambaBlock(nn.Module):
    """Simple Mamba block with layer normalization and residual connections"""
    def __init__(self, d_model, d_state, d_conv, expand, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.mamba = MambaSSM(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.dropout = nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff_dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Mamba block with residual
        residual = x
        x = self.layer_norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        x = x + residual

        # Feed-forward with residual
        residual = x
        x = self.ff_norm(x)
        x = self.feed_forward(x)
        x = self.ff_dropout(x)
        x = x + residual

        return x


class LSTMBlock(nn.Module):
    """LSTM block for temporal processing with layer normalization"""
    def __init__(self, d_model, hidden_size=None, num_layers=2, dropout=0.1):
        super().__init__()
        if hidden_size is None:
            hidden_size = d_model

        self.layer_norm = nn.LayerNorm(d_model)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)

        # Projection layer to match d_model output dimension
        self.output_proj = nn.Linear(hidden_size * 2, d_model)  # *2 for bidirectional

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        residual = x

        # Apply layer normalization
        x = self.layer_norm(x)

        # Process through LSTM
        lstm_out, _ = self.lstm(x)

        # Project to match input dimension
        x = self.output_proj(lstm_out)
        x = self.dropout(x)

        # Add residual connection
        x = x + residual

        return x


class Mamba4Traffic(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        # Get data features first to ensure num_nodes is defined
        self._scaler = self.data_feature.get('scaler')
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 1)
        self.output_dim = self.data_feature.get('output_dim', 1)

        # Get model config
        self.input_window = config.get('input_window', 12)
        self.output_window = config.get('output_window', 12)
        self.device = config.get('device', torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        self._logger = getLogger()

        # Ablation study flags
        self.no_mamba = config.get('noMamba', False)
        self.no_spatial = config.get('noSpatial', False)
        self.no_temporal = config.get('noTemporal', False)
        self.no_mamba_temporal = config.get('noMTemporal', False)
        self.no_lstm = config.get('noLSTM', False)
        self.no_embeddings = config.get('noEmbeddings', False)
        self.no_emb_spatial = config.get('noEmbSpatial', False)
        self.no_emb_temporal = config.get('noEmbTemporal', False)
        self.no_emb_adaptive = config.get('noEmbAdaptive', False)

        # Add time handling parameters (only if temporal embeddings are enabled)
        self.add_time_in_day = config.get("add_time_in_day", False) and not self.no_emb_temporal
        self.add_day_in_week = config.get("add_day_in_week", False) and not self.no_emb_temporal
        self.steps_per_day = config.get("steps_per_day", 288)

        # Get embedding dimensions from config
        self.input_embedding_dim = config.get('input_embedding_dim', 24)
        self.temporal_emb_dim = config.get('temporal_emb_dim', 24) if not self.no_emb_temporal else 0  # For time embeddings
        self.spatial_embedding_dim = config.get('spatial_embedding_dim', 16) if not self.no_emb_spatial else 0
        self.adaptive_embedding_dim = config.get('adaptive_embedding_dim', 80) if not self.no_emb_adaptive else 0

        # Get Mamba-specific parameters from config (define d_model first)
        self.d_model = config.get('d_model', 96)
        self.d_state = config.get('d_state', 32)
        self.d_conv = config.get('d_conv', 4)
        self.expand = config.get('expand', 2)
        self.dropout = config.get('dropout', 0.1)

        # Calculate total embedding dimensions for both paths
        if self.no_embeddings:
            self.total_embedding_dim = 0
        else:
            self.total_embedding_dim = (
                self.spatial_embedding_dim +
                self.adaptive_embedding_dim +
                (2 * self.temporal_emb_dim if self.add_time_in_day and self.add_day_in_week else
                 self.temporal_emb_dim if self.add_time_in_day or self.add_day_in_week else 0)
            )

        # Create embeddings
        self.input_proj = nn.Linear(self.feature_dim, self.d_model)  # Direct projection to d_model

        # Time embeddings for both paths (only if not disabled)
        if self.add_time_in_day and not self.no_embeddings:
            self.tod_embedding = nn.Embedding(self.steps_per_day, self.temporal_emb_dim)
        if self.add_day_in_week and not self.no_embeddings:
            self.dow_embedding = nn.Embedding(7, self.temporal_emb_dim)

        # Initialize spatial embedding (only if not disabled)
        if self.spatial_embedding_dim > 0 and not self.no_embeddings:
            self.spatial_embedding = nn.Parameter(torch.empty(self.num_nodes, self.spatial_embedding_dim))
            nn.init.xavier_uniform_(self.spatial_embedding)

        # Initialize adaptive embedding (only if not disabled)
        if self.adaptive_embedding_dim > 0 and not self.no_embeddings:
            self.adaptive_embedding = nn.Parameter(
                torch.empty(self.input_window, self.num_nodes, self.adaptive_embedding_dim)
            )
            nn.init.xavier_uniform_(self.adaptive_embedding)

        # LSTM-specific parameters
        self.lstm_hidden_size = config.get('lstm_hidden_size', self.d_model)
        self.lstm_num_layers = config.get('lstm_num_layers', 2)

        # Positional Encoding (learned) - for temporal path (only if temporal path exists)
        if not self.no_temporal:
            self.pos_encoding = nn.Parameter(torch.randn(1, self.input_window, 1, self.d_model))

        # FiLM: Modulate with MLP for temporal path embeddings (only if temporal path exists and embeddings are enabled)
        if not self.no_temporal and not self.no_embeddings and self.total_embedding_dim > 0:
            self.temporal_modulation_mlp = nn.Sequential(
                nn.Linear(self.total_embedding_dim, 2 * self.d_model),
                nn.ReLU(),
                nn.Linear(2 * self.d_model, 2 * self.d_model)
            )

        # FiLM: Modulate with MLP for spatial path embeddings (only if spatial path exists and embeddings are enabled)
        if not self.no_spatial and not self.no_embeddings and self.total_embedding_dim > 0:
            self.spatial_modulation_mlp = nn.Sequential(
                nn.Linear(self.total_embedding_dim, 2 * self.d_model),
                nn.ReLU(),
                nn.Linear(2 * self.d_model, 2 * self.d_model)
            )

        # Temporal modules (only if temporal path is enabled)
        if not self.no_temporal:
            if not self.no_mamba_temporal and not self.no_mamba:
                self.temporal_mamba_block = SimpleMambaBlock(
                    d_model=self.d_model,
                    d_state=self.d_state,
                    d_conv=self.d_conv,
                    expand=self.expand,
                    dropout=self.dropout
                )

            if not self.no_lstm:
                self.temporal_lstm_block = LSTMBlock(
                    d_model=self.d_model,
                    hidden_size=self.lstm_hidden_size,
                    num_layers=self.lstm_num_layers,
                    dropout=self.dropout
                )

        # Spatial processing block (Mamba) (only if spatial path is enabled)
        if not self.no_spatial and not self.no_mamba:
            self.spatial_block = SimpleMambaBlock(
                d_model=self.d_model,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                dropout=self.dropout
            )

        # Final fusion weights for temporal and spatial paths (only if both paths exist)
        if not self.no_temporal and not self.no_spatial:
            self.fusion_weights = nn.Parameter(torch.tensor([1.0, 1.0]))  # [temporal, spatial]
        elif not self.no_temporal:
            self.fusion_weights = nn.Parameter(torch.tensor([1.0]))  # [temporal]
        elif not self.no_spatial:
            self.fusion_weights = nn.Parameter(torch.tensor([1.0]))  # [spatial]

        # Temporal fusion weights for Mamba and LSTM outputs (only if both exist)
        if not self.no_temporal and not self.no_mamba_temporal and not self.no_mamba and not self.no_lstm:
            self.temporal_fusion_weights = nn.Parameter(torch.tensor([1.0, 1.0]))  # [mamba, lstm]
        elif not self.no_temporal and not self.no_mamba_temporal and not self.no_mamba:
            self.temporal_fusion_weights = nn.Parameter(torch.tensor([1.0]))  # [mamba]
        elif not self.no_temporal and not self.no_lstm:
            self.temporal_fusion_weights = nn.Parameter(torch.tensor([1.0]))  # [lstm]

        # Output projection layer
        self.output_proj = nn.Linear(self.d_model, self.output_dim)

        # Final layer normalization
        self.final_layer_norm = nn.LayerNorm(self.d_model)

        # Log the device being used and ablation configuration
        ablation_info = []
        if self.no_mamba:
            ablation_info.append("noMamba")
        if self.no_spatial:
            ablation_info.append("noSpatial")
        if self.no_temporal:
            ablation_info.append("noTemporal")
        if self.no_mamba_temporal:
            ablation_info.append("noMTemporal")
        if self.no_lstm:
            ablation_info.append("noLSTM")
        if self.no_embeddings:
            ablation_info.append("noEmbeddings")
        if self.no_emb_spatial:
            ablation_info.append("noEmbSpatial")
        if self.no_emb_temporal:
            ablation_info.append("noEmbTemporal")
        if self.no_emb_adaptive:
            ablation_info.append("noEmbAdaptive")

        ablation_str = f" (Ablations: {', '.join(ablation_info)})" if ablation_info else " (Full model)"
        self._logger.info(f"MCSTMamba ablation model configured for device: {self.device}{ablation_str}")

        # Move model to device
        self.to(self.device)

    def _prepare_full_embeddings(self, batch_size):
        """Prepare full embeddings (spatial, temporal, adaptive) for both paths"""
        # If no embeddings are enabled, return zeros
        if self.no_embeddings or self.total_embedding_dim == 0:
            return torch.zeros(batch_size, self.input_window, self.num_nodes, 0, device=self.device)

        # Create a single features tensor and add all features to it
        all_features = []

        # 1. Add temporal embeddings (time-of-day and day-of-week)
        if self.add_time_in_day and hasattr(self, 'tod_embedding'):
            tod = torch.linspace(0, 0.99, self.input_window, device=self.device)
            tod_indices = (tod * self.steps_per_day).long().clamp(0, self.steps_per_day - 1)
            tod_emb = self.tod_embedding(tod_indices).unsqueeze(0).unsqueeze(2)  # [1, T, 1, d_emb]
            all_features.append(tod_emb.expand(batch_size, self.input_window, self.num_nodes, -1))

        if self.add_day_in_week and hasattr(self, 'dow_embedding'):
            dow_indices = (torch.arange(self.input_window, device=self.device) % 7).long()
            dow_emb = self.dow_embedding(dow_indices).unsqueeze(0).unsqueeze(2)
            all_features.append(dow_emb.expand(batch_size, self.input_window, self.num_nodes, -1))

        # 2. Add spatial embeddings
        if hasattr(self, 'spatial_embedding'):
            spatial_emb = self.spatial_embedding.unsqueeze(0).unsqueeze(0)  # [1, 1, num_nodes, spatial_dim]
            spatial_emb = spatial_emb.expand(batch_size, self.input_window, -1, -1)
            all_features.append(spatial_emb)

        # 3. Add adaptive embeddings
        if hasattr(self, 'adaptive_embedding') and self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.unsqueeze(0)  # [1, input_window, num_nodes, adaptive_dim]
            adp_emb = adp_emb.expand(batch_size, -1, -1, -1)
            all_features.append(adp_emb)

        # Concatenate all embeddings into a single tensor
        if all_features:
            full_embeddings = torch.cat(all_features, dim=-1)  # [B, T, N, total_embedding_dim]
        else:
            full_embeddings = torch.zeros(batch_size, self.input_window, self.num_nodes, 0, device=self.device)

        return full_embeddings

    def forward(self, batch):
        # Move input to device if needed
        x = batch['X'].to(self.device)  # [batch_size, input_window, num_nodes, feature_dim]
        batch_size = x.shape[0]

        # Unified projection for both paths
        x_projected = self.input_proj(x)  # [batch_size, input_window, num_nodes, d_model]

        # Handle no Mamba case - direct output
        if self.no_mamba:
            x_out = self.final_layer_norm(x_projected)
            x_out = self.output_proj(x_out)
            return x_out[:, -self.output_window:]

        # Prepare full embeddings for both paths
        full_embeddings = self._prepare_full_embeddings(batch_size)  # [B, T, N, total_embedding_dim]

        # Initialize output tensors
        x_temporal_processed = None
        x_spatial_processed = None

        # Temporal path processing
        if not self.no_temporal:
            x_temporal = x_projected.clone()  # [batch_size, input_window, num_nodes, d_model]

            # Apply FiLM modulation for temporal path (if embeddings exist)
            if hasattr(self, 'temporal_modulation_mlp') and self.total_embedding_dim > 0:
                e_temporal_flat = full_embeddings.reshape(-1, self.total_embedding_dim)
                gamma_beta_temporal = self.temporal_modulation_mlp(e_temporal_flat)  # [B*T*N, 2*d_model]
                gamma_beta_temporal = gamma_beta_temporal.reshape(batch_size, self.input_window, self.num_nodes, 2 * self.d_model)
                gamma_temporal, beta_temporal = gamma_beta_temporal.chunk(2, dim=-1)
                x_temporal = gamma_temporal * x_temporal + beta_temporal  # FiLM-style modulation

            # Add Positional Encoding for temporal path
            if hasattr(self, 'pos_encoding'):
                x_temporal = x_temporal + self.pos_encoding  # [1, T, 1, d_model]

            # Reshape for temporal modules: [N, B*T, d_model]
            x_temp = x_temporal.permute(2, 0, 1, 3).reshape(self.num_nodes, batch_size * self.input_window, -1)

            # Process temporal modules
            temporal_outputs = []

            # Mamba temporal processing
            if hasattr(self, 'temporal_mamba_block'):
                x_mamba = self.temporal_mamba_block(x_temp)  # [N, B*T, d_model]
                temporal_outputs.append(x_mamba)

            # LSTM temporal processing
            if hasattr(self, 'temporal_lstm_block'):
                x_lstm = self.temporal_lstm_block(x_temp)  # [N, B*T, d_model]
                temporal_outputs.append(x_lstm)

            # Combine temporal outputs
            if len(temporal_outputs) == 1:
                x_temporal_processed = temporal_outputs[0]
            elif len(temporal_outputs) == 2:
                temporal_weights = F.softmax(self.temporal_fusion_weights, dim=0)
                x_temporal_processed = temporal_weights[0] * temporal_outputs[0] + temporal_weights[1] * temporal_outputs[1]
            else:
                x_temporal_processed = x_temp  # No temporal processing, use original

            # Reshape back: [B, T, N, d_model]
            x_temporal_processed = x_temporal_processed.reshape(self.num_nodes, batch_size, self.input_window, -1).permute(1, 2, 0, 3)

        # Spatial path processing
        if not self.no_spatial:
            x_spatial = x_projected.clone()  # [batch_size, input_window, num_nodes, d_model]

            # Apply FiLM modulation for spatial path (if embeddings exist)
            if hasattr(self, 'spatial_modulation_mlp') and self.total_embedding_dim > 0:
                e_spatial_flat = full_embeddings.reshape(-1, self.total_embedding_dim)
                gamma_beta_spatial = self.spatial_modulation_mlp(e_spatial_flat)  # [B*T*N, 2*d_model]
                gamma_beta_spatial = gamma_beta_spatial.reshape(batch_size, self.input_window, self.num_nodes, 2 * self.d_model)
                gamma_spatial, beta_spatial = gamma_beta_spatial.chunk(2, dim=-1)
                x_spatial = gamma_spatial * x_spatial + beta_spatial  # FiLM-style modulation

            # Spatial processing
            if hasattr(self, 'spatial_block'):
                is_large_dataset = self.num_nodes > 300

                if is_large_dataset and batch_size > 1:
                    # For large datasets, optimize with GPU-efficient chunking
                    x_spatial_processed = torch.zeros(batch_size, self.input_window, self.num_nodes, self.d_model,
                                            device=self.device)

                    # Process each timestep
                    for t in range(self.input_window):
                        nodes_seq = x_spatial[:, t, :, :]
                        effective_batch_size = 1

                        all_results = []
                        for b_idx in range(0, batch_size, effective_batch_size):
                            end_idx = min(b_idx + effective_batch_size, batch_size)
                            batch_slice = nodes_seq[b_idx:end_idx]
                            spatial_hidden = self.spatial_block(batch_slice)
                            all_results.append(spatial_hidden)

                        x_spatial_processed[:, t] = torch.cat(all_results, dim=0)
                else:
                    # For smaller datasets, process all at once
                    x_spatial_processed = x_spatial.permute(1, 0, 2, 3)  # [input_window, batch_size, num_nodes, d_model]
                    x_spatial_processed = x_spatial_processed.reshape(self.input_window, batch_size * self.num_nodes, -1)

                    # Process through spatial block
                    x_spatial_processed = self.spatial_block(x_spatial_processed)

                    # Reshape back to [batch_size, input_window, num_nodes, d_model]
                    x_spatial_processed = x_spatial_processed.reshape(self.input_window, batch_size, self.num_nodes, self.d_model)
                    x_spatial_processed = x_spatial_processed.permute(1, 0, 2, 3)  # [batch_size, input_window, num_nodes, d_model]
            else:
                x_spatial_processed = x_spatial  # No spatial processing, use original

        # Combine paths
        if x_temporal_processed is not None and x_spatial_processed is not None:
            # Both paths exist
            x_temp_norm = F.layer_norm(x_temporal_processed, x_temporal_processed.shape[-1:])
            x_spatial_norm = F.layer_norm(x_spatial_processed, x_spatial_processed.shape[-1:])

            if hasattr(self, 'fusion_weights') and len(self.fusion_weights) == 2:
                weights = F.softmax(self.fusion_weights, dim=0)
                x_combined = weights[0] * x_temp_norm + weights[1] * x_spatial_norm
            else:
                x_combined = (x_temp_norm + x_spatial_norm) / 2
        elif x_temporal_processed is not None:
            # Only temporal path exists
            x_combined = F.layer_norm(x_temporal_processed, x_temporal_processed.shape[-1:])
        elif x_spatial_processed is not None:
            # Only spatial path exists
            x_combined = F.layer_norm(x_spatial_processed, x_spatial_processed.shape[-1:])
        else:
            # No processing paths, use original projection
            x_combined = x_projected

        # Final processing and output projection
        x_out = self.final_layer_norm(x_combined)
        x_out = self.output_proj(x_out)

        return x_out[:, -self.output_window:]  # Return last output_window steps

    def calculate_loss(self, batch):
        """
        Calculate the training loss for a batch of data
        :param batch: Input data dictionary
        :return: Training loss (tensor)
        """
        y_true = batch['y'].to(self.device)
        y_predicted = self.predict(batch)

        # Apply inverse normalization
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])

        # Calculate masked MAE loss
        return loss.masked_mae_torch(y_predicted, y_true, 0)

    def predict(self, batch):
        """
        Make predictions for a batch of data
        :param batch: Input data dictionary
        :return: Predictions with shape [batch_size, output_window, num_nodes, output_dim]
        """
        return self.forward(batch)
