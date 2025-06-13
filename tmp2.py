class GuidedWrapper(nn.Module):
    """
    Universal wrapper for DiT or SiT that injects a learnable guidance scale `w`.
    This wrapper adds w_emb only if `w` is provided.
    """
    def __init__(self, base_model, w_dim=1, embed_dim=1152):
        super().__init__()
        self.base_model = base_model
        self.embed_dim = embed_dim

        self.w_embed = nn.Sequential(
            nn.Linear(w_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        #self.w_embed[-1].weight.data.zero_()
        #self.w_embed[-1].bias.data.zero_()

    def forward(self, x, t, y, w=None):
        # Embed timestep and label
        t_emb = self.base_model.t_embedder(t)                 # (B, D)
        y_emb = self.base_model.y_embedder(y, self.training)  # (B, D)

        # Inject guidance if available
        if w is not None:
            w_emb = self.w_embed(w-1)                           # (B, D)
            cond_std = (t_emb + y_emb).std(dim=-1, keepdim=True).detach()  # (B, 1)
            # 3. Apply manual scaling
            w_emb = w_emb * cond_std * 0.5  # 0.5 is a tunable dampening factor
            
            # print("_____________________________________________________")
            #print(f"[DEBUG] w_emb mean: {w_emb.mean().item():.4f}, std: {w_emb.std().item():.4f}")
            
            # print(w_emb)
            # print(f"[DEBUG] w_emb mean: {w_emb.mean().item():.4f}, std: {w_emb.std().item():.4f}")
            # print(f"[DEBUG] t_emb mean: {t_emb.mean().item():.4f}, std: {t_emb.std().item():.4f}")
            # print(f"[DEBUG] y_emb mean: {y_emb.mean().item():.4f}, std: {y_emb.std().item():.4f}")
            c = t_emb + y_emb + w_emb

        else:
            c = t_emb + y_emb

        # Run through the backbone
        x = self.base_model.x_embedder(x) + self.base_model.pos_embed  # (B, T, D)
        for block in self.base_model.blocks:
            x = block(x, c)                                   # (B, T, D)
        x = self.base_model.final_layer(x, c)                 # (B, T, patch^2 * C)
        return self.base_model.unpatchify(x)                  # (B, out_channels, H, W)

    def __getattr__(self, name):
        # Only forward attribute access if not found on wrapper itself
        if name in self.__dict__:
            return self.__dict__[name]
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)