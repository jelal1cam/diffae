import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def get_timestep_embedding(timesteps, embedding_dim, max_period=10000):
    """Create sinusoidal timestep embeddings."""
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(0, half_dim, device=timesteps.device, dtype=torch.float32) / half_dim
    emb = timesteps.float().unsqueeze(1) * torch.exp(exponent.unsqueeze(0))
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


# --------- Original Architecture (Preserved Exactly) ---------

class LinearTimeDependentClassifier(nn.Module):
    """
    A linear classifier with time conditioning.
    """
    
    def __init__(self, in_features, num_classes, time_embedding_dim):
        """
        Args:
            in_features (int): Dimensionality of the feature input.
            num_classes (int): Number of output classes.
            time_embedding_dim (int): Dimensionality of the time embedding.
        """
        super().__init__()
        self.time_embedding_dim = time_embedding_dim
        
        # MLP to process the sinusoidal time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(time_embedding_dim, time_embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(time_embedding_dim, time_embedding_dim)
        )
        
        # Linear layer that operates on the concatenated vector (feature + time_embedding)
        self.linear = nn.Linear(in_features + time_embedding_dim, num_classes)
    
    def forward(self, x, t=None):
        if t is None:
            t = torch.zeros(x.size(0), device=x.device, dtype=torch.float32)
        t_emb = get_timestep_embedding(t, self.time_embedding_dim)
        t_emb = self.time_embed(t_emb)
        x_cat = torch.cat([x, t_emb], dim=-1)
        return self.linear(x_cat)


class FlexibleClassifier(nn.Module):
    """
    Original FlexibleClassifier for fallback/comparison.
    """
    
    def __init__(self, in_features, num_classes, hidden_dims=[], dropout=0.25, time_embedding_dim=None):
        super().__init__()
        self.use_time = time_embedding_dim is not None
        
        if self.use_time:
            self.time_embedding_dim = time_embedding_dim
            self.time_embed = nn.Sequential(
                nn.Linear(self.time_embedding_dim, self.time_embedding_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.time_embedding_dim, self.time_embedding_dim)
            )
            effective_in_features = in_features + self.time_embedding_dim
        else:
            effective_in_features = in_features

        # Build the main network
        layers = []
        prev_dim = effective_in_features
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.LayerNorm(dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x, t=None):
        if self.use_time:
            if t is None:
                t = torch.zeros(x.size(0), device=x.device, dtype=torch.float32)
            t_emb = get_timestep_embedding(t, self.time_embedding_dim)
            t_emb = self.time_embed(t_emb)
            x = torch.cat([x, t_emb], dim=-1)
        return self.model(x)


# --------- Enhanced Architecture (New Implementations) ---------

class EnhancedTimeEmbed(nn.Module):
    """Enhanced time embedding with wider intermediate layer and optional SiLU activation."""
    def __init__(self, time_embedding_dim, expansion_factor=2, use_silu=True):
        super().__init__()
        act_fn = nn.SiLU(inplace=True) if use_silu else nn.ReLU(inplace=True)
        self.net = nn.Sequential(
            nn.Linear(time_embedding_dim, time_embedding_dim * expansion_factor),
            act_fn,
            nn.Linear(time_embedding_dim * expansion_factor, time_embedding_dim)
        )
    
    def forward(self, t_emb):
        return self.net(t_emb)


class ResidualBlock(nn.Module):
    """Residual block with pre-norm architecture."""
    def __init__(self, dim, dropout=0.0, use_silu=True):
        super().__init__()
        act_fn = nn.SiLU(inplace=True) if use_silu else nn.ReLU(inplace=True)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            act_fn,
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
    
    def forward(self, x):
        return x + self.net(x)


class EnhancedFlexibleClassifier(nn.Module):
    """Enhanced flexible classifier with residual connections and improved time embedding."""
    
    def __init__(self, in_features, num_classes, hidden_dims=[], dropout=0.25, 
                 time_embedding_dim=None, use_silu=True, use_residual=True):
        super().__init__()
        self.use_time = time_embedding_dim is not None
        
        # Choose activation
        self.use_silu = use_silu
        self.use_residual = use_residual
        act_fn = nn.SiLU(inplace=True) if use_silu else nn.ReLU(inplace=True)
        
        if self.use_time:
            self.time_embedding_dim = time_embedding_dim
            self.time_embed = EnhancedTimeEmbed(time_embedding_dim, use_silu=use_silu)
            effective_in_features = in_features + self.time_embedding_dim
        else:
            effective_in_features = in_features

        # Build the main network
        layers = []
        prev_dim = effective_in_features
        
        for i, dim in enumerate(hidden_dims):
            # Check if we can use residual connection
            if use_residual and prev_dim == dim and i > 0:
                layers.append(ResidualBlock(dim, dropout, use_silu))
            else:
                # Standard layer with projection
                layers.append(nn.Linear(prev_dim, dim))
                layers.append(nn.LayerNorm(dim))
                if use_silu:
                    layers.append(nn.SiLU(inplace=True))
                else:
                    layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(dropout))
            prev_dim = dim
            
        # Output layer with optional scale
        layers.append(nn.Linear(prev_dim, num_classes))
        self.model = nn.Sequential(*layers)
        self.scale = nn.Parameter(torch.ones(1))
    
    def forward(self, x, t=None):
        if self.use_time:
            if t is None:
                t = torch.zeros(x.size(0), device=x.device, dtype=torch.float32)
            t_emb = get_timestep_embedding(t, self.time_embedding_dim)
            t_emb = self.time_embed(t_emb)
            x = torch.cat([x, t_emb], dim=-1)
        return self.model(x) * self.scale


class FiLMResidualBlock(nn.Module):
    """
    A pre-norm residual block that uses FiLM: we generate scale&shift
    from the time embedding and apply it to the normalized features.
    
    This improved version handles dimension changes between blocks.
    """
    def __init__(self, in_dim, out_dim, time_dim, dropout=0.2, use_silu=True):
        super().__init__()
        act = nn.SiLU(inplace=True) if use_silu else nn.ReLU(inplace=True)
        
        # from t_emb to 2*out_dim FiLM params
        self.film = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            act,
            nn.Linear(time_dim, 2 * out_dim),
        )
        
        self.norm = nn.LayerNorm(in_dim)
        
        # Feedforward with dimension handling
        self.ff = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            act,
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )
        
        # Projection layer for residual connection if dimensions don't match
        self.needs_projection = in_dim != out_dim
        if self.needs_projection:
            self.proj = nn.Linear(in_dim, out_dim, bias=False)
            
    def forward(self, x, t_emb):
        # Save the input for residual connection
        identity = x
        
        # Normalization and FiLM conditioning
        h = self.norm(x)
        
        # Apply FiLM (feature-wise linear modulation)
        gammas, betas = self.film(t_emb).chunk(2, dim=-1)
        
        # Apply feedforward layers (including dimension change)
        h = self.ff(h)
        
        # Apply modulation after dimension change
        h = gammas * h + betas
        
        # Apply residual connection with projection if needed
        if self.needs_projection:
            return self.proj(identity) + h
        else:
            return identity + h


class FiLMClassifier(nn.Module):
    """
    Deeper, FiLM-conditioned MLP.
    - embeds time with EnhancedTimeEmbed
    - first projects input→hidden
    - stacks FiLMResidualBlocks with proper dimension handling
    - final read-out to num_classes
    """
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_dims=[512, 256],
        dropout=0.3,
        time_embedding_dim=64,
        use_silu=True,
    ):
        super().__init__()
        # time embed network (wider internals)
        self.time_embed = EnhancedTimeEmbed(time_embedding_dim, use_silu=use_silu)
        
        # initial projection
        self.input_proj = nn.Linear(in_features + time_embedding_dim, hidden_dims[0])
        act = nn.SiLU(inplace=True) if use_silu else nn.ReLU(inplace=True)
        self.input_norm = nn.LayerNorm(hidden_dims[0])
        
        # build a stack of FiLMResidualBlocks with proper dimension handling
        blocks = []
        for i in range(len(hidden_dims)):
            in_dim = hidden_dims[i]
            # If last layer, maintain same dimension for proper residual
            out_dim = hidden_dims[i] if i == len(hidden_dims) - 1 else hidden_dims[i+1]
            blocks.append(FiLMResidualBlock(
                in_dim=in_dim, 
                out_dim=out_dim, 
                time_dim=time_embedding_dim, 
                dropout=dropout, 
                use_silu=use_silu
            ))
        
        self.blocks = nn.ModuleList(blocks)
        
        # final classifier head
        self.head = nn.Linear(hidden_dims[-1], num_classes)
        
        # global scale parameter
        self.scale = nn.Parameter(torch.ones(1))
    
    def forward(self, x, t=None):
        # x: [B, in_features], t: [B] timesteps
        if t is None:
            t = torch.zeros(x.size(0), device=x.device, dtype=torch.float32)
            
        # Get time embedding
        t_emb = get_timestep_embedding(t, self.time_embed.net[0].in_features)
        t_emb = self.time_embed(t_emb)  # [B, time_embedding_dim]
        
        # concat and project
        x = torch.cat([x, t_emb], dim=-1)
        x = self.input_norm(self.input_proj(x))
        
        # Process through residual blocks
        for i, block in enumerate(self.blocks):
            x = block(x, t_emb)
            
        # Final classification head
        return self.head(x) * self.scale

# --------- Unified Interface ---------

def build_classifier(conf, input_dim, num_classes):
    """
    Build classifier based on classifier_type.
    
    Args:
        conf: Configuration object with classifier settings
    
    Returns:
        A classifier instance
    """
    
    # Get classifier type - this is the main selector for architecture
    classifier_type = getattr(conf, 'classifier_type', 'linear')
    
    # Time dependence is a parameter of the classifier, not a separate choice
    time_dependent = getattr(conf, 'diffusion_time_dependent_classifier', False)
    time_embedding_dim = getattr(conf, 'time_embedding_dim', 64) if time_dependent else None
    
    # Create the appropriate classifier based on type
    if classifier_type == 'linear':
        if time_dependent:
            classifier = LinearTimeDependentClassifier(
                in_features=input_dim,
                num_classes=num_classes,
                time_embedding_dim=time_embedding_dim
            )
        else:
            classifier = nn.Linear(input_dim, num_classes)
            
    elif classifier_type == 'flexible':
        hidden_dims = getattr(conf, 'non_linear_hidden_dims', [])
        dropout = getattr(conf, 'non_linear_dropout', 0.2)
        
        classifier = FlexibleClassifier(
            in_features=input_dim,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            dropout=dropout,
            time_embedding_dim=time_embedding_dim
        )
        
    elif classifier_type == 'enhanced':
        hidden_dims = getattr(conf, 'non_linear_hidden_dims', [])
        dropout = getattr(conf, 'non_linear_dropout', 0.2)
        use_silu = getattr(conf, 'use_silu_activation', True)
        use_residual = getattr(conf, 'use_residual_connections', True)
        
        classifier = EnhancedFlexibleClassifier(
            in_features=input_dim,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            dropout=dropout,
            time_embedding_dim=time_embedding_dim,
            use_silu=use_silu,
            use_residual=use_residual
        )
        
    elif classifier_type == 'film':
        hidden_dims = getattr(conf, 'non_linear_hidden_dims', [512, 256])
        dropout = getattr(conf, 'non_linear_dropout', 0.3)
        use_silu = getattr(conf, 'use_silu_activation', True)
        
        if not time_dependent:
            raise ValueError("FiLM classifier requires time dependence (diffusion_time_dependent_classifier=True)")
            
        classifier = FiLMClassifier(
            in_features=input_dim,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            dropout=dropout,
            time_embedding_dim=time_embedding_dim,
            use_silu=use_silu
        )
        
    else:
        raise ValueError(f"Unknown classifier_type: {classifier_type}")
    
    return classifier


def get_classifier_info(conf):
    """
    Get classifier configuration info.
    
    Args:
        conf: Configuration object with classifier settings
    
    Returns:
        A dictionary with classifier information
    """
    # Extract common parameters
    input_dim = getattr(conf, 'input_dim', 0)
    num_classes = getattr(conf, 'num_classes', 0)
    classifier_type = getattr(conf, 'classifier_type', 'linear')
    time_dependent = getattr(conf, 'diffusion_time_dependent_classifier', False)
    
    # Create basic info dictionary
    info = {
        'input_dimensions': input_dim,
        'output_classes': num_classes,
        'classifier_type': classifier_type,
        'time_dependent': time_dependent
    }
    
    # Add time embedding info if needed
    if time_dependent:
        info['time_embedding_dim'] = getattr(conf, 'time_embedding_dim', 64)
    
    # Add classifier-specific details
    if classifier_type in ['flexible', 'enhanced', 'film']:
        info['hidden_dims'] = getattr(conf, 'non_linear_hidden_dims', [])
        info['dropout'] = getattr(conf, 'non_linear_dropout', 0.2)
        
        if classifier_type in ['enhanced', 'film']:
            info['use_silu_activation'] = getattr(conf, 'use_silu_activation', True)
            
        if classifier_type == 'enhanced':
            info['use_residual_connections'] = getattr(conf, 'use_residual_connections', True)
    
    # Create a human-readable description
    descriptions = {
        'linear': "Linear classifier",
        'flexible': "Flexible MLP classifier",
        'enhanced': "Enhanced flexible classifier with residual connections",
        'film': "FiLM-conditioned MLP classifier with feature modulation"
    }
    
    info['description'] = descriptions.get(classifier_type, "Unknown classifier")
    if time_dependent:
        info['description'] += " (time-dependent)"
    
    return info


def print_classifier_info(conf):
    """
    Print detailed classifier configuration in a nice formatted way.
    
    Args:
        conf: Configuration object with classifier settings
    """
    info = get_classifier_info(conf)
    
    print("=" * 50)
    print(f"Classifier: {info['description']}")
    print("=" * 50)
    
    print(f"Type: {info['classifier_type']}")
    print(f"Input Dimensions: {info['input_dimensions']}")
    print(f"Output Classes: {info['output_classes']}")
    print(f"Time-Dependent: {info['time_dependent']}")
    
    if info['time_dependent']:
        print(f"Time Embedding Dim: {info['time_embedding_dim']}")
    
    if 'hidden_dims' in info:
        print(f"Hidden Dimensions: {info['hidden_dims']}")
        print(f"Dropout Rate: {info['dropout']}")
    
    if 'use_silu_activation' in info:
        print(f"Activation: {'SiLU' if info['use_silu_activation'] else 'ReLU'}")
    
    if 'use_residual_connections' in info:
        print(f"Residual Connections: {'Enabled' if info['use_residual_connections'] else 'Disabled'}")
    
    print("=" * 50)