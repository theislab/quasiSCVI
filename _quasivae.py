from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Callable, Literal, Optional

import numpy as np
import torch
from torch.distributions import Distribution
from torch.nn.functional import one_hot
import torch.nn.functional as F


from scvi import REGISTRY_KEYS
# from scvi.module._constants import MODULE_KEYS
from _constants import MODULE_KEYS, EXTRA_KEYS

from scvi.nn import DecoderSCVI, Encoder

llogger = logging.getLogger(__name__)
from scvi.module.base import (
    BaseMinifiedModeModuleClass,
    EmbeddingModuleMixin,
    LossOutput,
    auto_move_data,
)

def quasi_likelihood_loss(px_rate, target, px_r, b):
    residual = torch.pow(target - px_rate, 2)
    b = torch.clamp(b, min=0, max=3)
    variance = px_r * torch.pow(px_rate, b)
    quasi_likelihood = residual / variance 
    return quasi_likelihood



class QuasiVAE(BaseMinifiedModeModuleClass, EmbeddingModuleMixin):
    def __init__(
        self,
        n_input: int,
        z_guide_m: torch.Tensor | None = None,
        z_guide_v: torch.Tensor | None = None,
        gbc_latent_dim: int | None = None,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_latent: int = 15,
        b_dim: int = 10,
        n_layers: int = 1,
        n_continuous_cov: int = 0,
        n_cats_per_cov: list[int] | None = None,
        dropout_rate: float = 0.1,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        log_variational: bool = True,
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
        latent_distribution: Literal["normal", "ln"] = "normal",
        encode_covariates: bool = False,
        deeply_inject_covariates: bool = True,
        batch_representation: Literal["one-hot", "embedding"] = "embedding",
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        use_size_factor_key: bool = False,
        use_observed_lib_size: bool = True,
        library_log_means: np.ndarray | None = None,
        library_log_vars: np.ndarray | None = None,
        var_activation: Callable[[torch.Tensor], torch.Tensor] = None,
        extra_encoder_kwargs: dict | None = None,
        extra_decoder_kwargs: dict | None = None,
        batch_embedding_kwargs: dict | None = None,
        b_prior_mixture: bool = False,
        b_prior_mixture_k: int = 5,
    ):

        super().__init__()
        self.dispersion = dispersion
        self.n_latent = n_latent
        self.b_dim = b_dim
        self.log_variational = log_variational
        self.gene_likelihood = gene_likelihood
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.latent_distribution = latent_distribution
        self.encode_covariates = encode_covariates
        self.use_size_factor_key = use_size_factor_key
        self.use_observed_lib_size = use_size_factor_key or use_observed_lib_size
        self.gbc_latent_dim = gbc_latent_dim
        self.kl_r_log = []  # List to store kl_r values

        if not self.use_observed_lib_size:
            if library_log_means is None or library_log_vars is None:
                raise ValueError(
                    "If not using observed_lib_size, "
                    "must provide library_log_means and library_log_vars."
                )

            self.register_buffer("library_log_means", torch.from_numpy(library_log_means).float())
            self.register_buffer("library_log_vars", torch.from_numpy(library_log_vars).float())
        
        self.z_guide_m = z_guide_m
        self.z_guide_v = z_guide_v
        self.px_b = torch.nn.Parameter(torch.full((n_input,), 2.0))
        self.batch_representation = batch_representation
        if self.batch_representation == "embedding":
            self.init_embedding(REGISTRY_KEYS.BATCH_KEY, n_batch, **(batch_embedding_kwargs or {}))
            batch_dim = self.get_embedding(REGISTRY_KEYS.BATCH_KEY).embedding_dim
        elif self.batch_representation != "one-hot":
            raise ValueError("`batch_representation` must be one of 'one-hot', 'embedding'.")

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"
        
        n_input_encoder = n_input + n_continuous_cov * encode_covariates
        if self.batch_representation == "embedding":
            n_input_encoder += batch_dim * encode_covariates
            cat_list = list([] if n_cats_per_cov is None else n_cats_per_cov)
        else:
            cat_list = [n_batch] + list([] if n_cats_per_cov is None else n_cats_per_cov)

        encoder_cat_list = cat_list if encode_covariates else None
        _extra_encoder_kwargs = extra_encoder_kwargs or {}

        # TO DO: this has to go from qzm
        if self.gbc_latent_dim is None:
                self.px_r_encoder = Encoder(
                n_latent,
                b_dim,
                n_layers=1,
                n_cat_list=encoder_cat_list,
                n_hidden=n_hidden,
                dropout_rate=dropout_rate,
                inject_covariates=deeply_inject_covariates,
                use_batch_norm=use_batch_norm_encoder,
                use_layer_norm=use_layer_norm_encoder,
                var_activation=var_activation,
                return_dist=True,
                **_extra_encoder_kwargs,
            )
        else:
            self.px_r_encoder = Encoder(
                n_latent,
                gbc_latent_dim,
                n_layers=1,
                n_cat_list=encoder_cat_list,
                n_hidden=n_hidden,
                dropout_rate=dropout_rate,
                inject_covariates=deeply_inject_covariates,
                use_batch_norm=use_batch_norm_encoder,
                use_layer_norm=use_layer_norm_encoder,
                var_activation=var_activation,
                return_dist=True,
                **_extra_encoder_kwargs,
            )
        self.z_encoder = Encoder(
            n_input_encoder,
            n_latent,
            n_cat_list=encoder_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            return_dist=False,
            **_extra_encoder_kwargs,
        )
        # l encoder goes from n_input-dimensional data to 1-d library size
        self.l_encoder = Encoder(
            n_input_encoder,
            1,
            n_layers=1,
            n_cat_list=encoder_cat_list,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            return_dist=True,
            **_extra_encoder_kwargs,
        )
        n_input_decoder = n_latent + n_continuous_cov
        if self.batch_representation == "embedding":
            n_input_decoder += batch_dim

        _extra_decoder_kwargs = extra_decoder_kwargs or {}
        self.decoder = DecoderSCVI(
            n_input_decoder,
            n_input,
            n_cat_list=cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
            scale_activation="softplus" if use_size_factor_key else "softmax",
            **_extra_decoder_kwargs,
        )

  
        if self.gbc_latent_dim is None:
            qrx = torch.nn.Linear(b_dim, 1)
        else:
            qrx = torch.nn.Linear(gbc_latent_dim, 1)
        
        # self.b_decoder = torch.nn.Sequential(
        #     torch.nn.Linear(b_dim, 1),  # Linear transformation
        #     #torch.nn.Softmax(dim=-1)              # Softmax activation
        # ) 

        self.px_r_decoder = torch.nn.Sequential(
        qrx,  # Linear transformation
        torch.nn.Softmax(dim=-1)   )           # Softmax activation
       
        self.b_prior_mixture = b_prior_mixture
        self.b_prior_mixture_k = b_prior_mixture_k
        if self.b_prior_mixture:
            if self.n_labels > 1:
                b_prior_mixture_k = self.n_labels
            else:
                b_prior_mixture_k = self.b_prior_mixture_k
        # Initialize parameters for the mixture model
            self.b_prior_logits = torch.nn.Parameter(torch.zeros(b_prior_mixture_k))
            self.b_prior_means = torch.nn.Parameter(torch.randn(b_dim, b_prior_mixture_k))
            self.b_prior_scales = torch.nn.Parameter(torch.zeros(b_dim, b_prior_mixture_k))



    def _get_inference_input(self,
        tensors: dict[str, torch.Tensor | None],
    ) -> dict[str, torch.Tensor | None]:
        """Get input tensors for the inference process."""
        from scvi.data._constants import ADATA_MINIFY_TYPE
                  
        if self.minified_data_type is None:
            return {
                MODULE_KEYS.X_KEY: tensors[REGISTRY_KEYS.X_KEY],
                MODULE_KEYS.BATCH_INDEX_KEY: tensors[REGISTRY_KEYS.BATCH_KEY],
                MODULE_KEYS.CONT_COVS_KEY: tensors.get(REGISTRY_KEYS.CONT_COVS_KEY, None),
                MODULE_KEYS.CAT_COVS_KEY: tensors.get(REGISTRY_KEYS.CAT_COVS_KEY, None),
                
            }
        elif self.minified_data_type == ADATA_MINIFY_TYPE.LATENT_POSTERIOR:
            return {
                MODULE_KEYS.QZM_KEY: tensors[REGISTRY_KEYS.LATENT_QZM_KEY],
                MODULE_KEYS.QZV_KEY: tensors[REGISTRY_KEYS.LATENT_QZV_KEY],
                REGISTRY_KEYS.OBSERVED_LIB_SIZE: tensors[REGISTRY_KEYS.OBSERVED_LIB_SIZE],
            }
        else:
            raise NotImplementedError(f"Unknown minified-data type: {self.minified_data_type}")

    def _get_generative_input( self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor | Distribution | None],
    ) -> dict[str, torch.Tensor | None]:
        """Get input tensors for the generative process."""
        size_factor = tensors.get(REGISTRY_KEYS.SIZE_FACTOR_KEY, None)
        if size_factor is not None:
            size_factor = torch.log(size_factor)
        
        return {
            MODULE_KEYS.Z_KEY: inference_outputs[MODULE_KEYS.Z_KEY],
            MODULE_KEYS.LIBRARY_KEY: inference_outputs[MODULE_KEYS.LIBRARY_KEY],
            MODULE_KEYS.BATCH_INDEX_KEY: tensors[REGISTRY_KEYS.BATCH_KEY],
            MODULE_KEYS.Y_KEY: tensors[REGISTRY_KEYS.LABELS_KEY],
            MODULE_KEYS.CONT_COVS_KEY: tensors.get(REGISTRY_KEYS.CONT_COVS_KEY, None),
            MODULE_KEYS.CAT_COVS_KEY: tensors.get(REGISTRY_KEYS.CAT_COVS_KEY, None),
            MODULE_KEYS.SIZE_FACTOR_KEY: size_factor,
            #MODULE_KEYS.B_KEY: tensors.get(EXTRA_KEYS.LATENT_QB_KEY, None),
            MODULE_KEYS.PX_R_KEY:  inference_outputs[MODULE_KEYS.PX_R_KEY],

            
        
        }

    def _compute_local_library_params(
        self,
        batch_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes local library parameters.

        Compute two tensors of shape (batch_index.shape[0], 1) where each
        element corresponds to the mean and variances, respectively, of the
        log library sizes in the batch the cell corresponds to.
        """
        from torch.nn.functional import linear

        n_batch = self.library_log_means.shape[1]
        local_library_log_means = linear(
            one_hot(batch_index.squeeze(-1), n_batch).float(), self.library_log_means
        )

        local_library_log_vars = linear(
            one_hot(batch_index.squeeze(-1), n_batch).float(), self.library_log_vars
        )

        return local_library_log_means, local_library_log_vars

    @auto_move_data
    def _regular_inference(
        self,
        x: torch.Tensor,
        batch_index: torch.Tensor,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
        n_samples: int = 1,
    ) -> dict[str, torch.Tensor | Distribution | None]:
        """Run the regular inference process."""
        
        from torch.distributions import Normal

        x_ = x
        if self.use_observed_lib_size:
            library = torch.log(x.sum(1)).unsqueeze(1)
        if self.log_variational:
            x_ = torch.log1p(x_)

        if cont_covs is not None and self.encode_covariates:
            encoder_input = torch.cat((x_, cont_covs), dim=-1)
        else:
            encoder_input = x_
        if cat_covs is not None and self.encode_covariates:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()

        
        if self.batch_representation == "embedding" and self.encode_covariates:
            batch_rep = self.compute_embedding(REGISTRY_KEYS.BATCH_KEY, batch_index)
            encoder_input = torch.cat([encoder_input, batch_rep], dim=-1)
            q_m, q_v, z = self.z_encoder(encoder_input, *categorical_input)
        else:
            q_m, q_v, z = self.z_encoder(encoder_input, batch_index, *categorical_input)
            
        qz= Normal(q_m, q_v.sqrt())
            
        qr = None
        if cont_covs is not None:
            px_r_encoder_input = torch.cat((q_m, cont_covs), dim=-1)

        else:
            px_r_encoder_input = q_m


        qr ,px_r = self.px_r_encoder(px_r_encoder_input) 
 
        ql = None
    
        if not self.use_observed_lib_size:
            if self.batch_representation == "embedding":
                ql, library_encoded = self.l_encoder(encoder_input, *categorical_input)
                
            else:
                ql, library_encoded = self.l_encoder(
                    encoder_input, batch_index, *categorical_input
                )
                
            library = library_encoded

        if n_samples > 1:
            untran_z = qz.sample((n_samples,))
            z = self.z_encoder.z_transformation(untran_z)
            
            if self.gbc_latent_dim is None:
                untran_r = qr.sample((n_samples,))
                px_r = self.px_r_encoder.z_transformation(untran_r)
            
            if self.use_observed_lib_size:
                library = library.unsqueeze(0).expand(
                    (n_samples, library.size(0), library.size(1))
                )
                
        
            else:
                library = ql.sample((n_samples,))
 
        
        return {
            MODULE_KEYS.Z_KEY: z,
            MODULE_KEYS.QZ_KEY: qz,
            MODULE_KEYS.QL_KEY: ql,
            MODULE_KEYS.LIBRARY_KEY: library,
            MODULE_KEYS.QR_KEY: qr,
            MODULE_KEYS.PX_R_KEY: px_r,
        }

    @auto_move_data
    def _cached_inference( 
        self,
        qzm: torch.Tensor,
        qzv: torch.Tensor,
        observed_lib_size: torch.Tensor,
        n_samples: int = 1,
    ) -> dict[str, torch.Tensor | None]:
        """Run the cached inference process."""
        from torch.distributions import Normal

        from scvi.data._constants import ADATA_MINIFY_TYPE

        if self.minified_data_type != ADATA_MINIFY_TYPE.LATENT_POSTERIOR:
            raise NotImplementedError(f"Unknown minified-data type: {self.minified_data_type}")

        dist = Normal(qzm, qzv.sqrt())
        # use dist.sample() rather than rsample because we aren't optimizing the z here
        untran_z = dist.sample() if n_samples == 1 else dist.sample((n_samples,))
        z = self.z_encoder.z_transformation(untran_z)

        library = torch.log(observed_lib_size)
        if n_samples > 1:
            library = library.unsqueeze(0).expand((n_samples, library.size(0), library.size(1)))

        return {
            MODULE_KEYS.Z_KEY: z,
            MODULE_KEYS.QZM_KEY: qzm,
            MODULE_KEYS.QZV_KEY: qzv,
            MODULE_KEYS.QL_KEY: None,
            MODULE_KEYS.LIBRARY_KEY: library,
        }

    @auto_move_data
    def generative(self,
        z: torch.Tensor,
        px_r: torch.Tensor, 
        library: torch.Tensor,
        batch_index: torch.Tensor,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
        size_factor: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        transform_batch: torch.Tensor | None = None,
    ) -> dict[str, Distribution | None]:
        """Run the generative process."""
        from torch.distributions import Normal
        from torch.nn.functional import linear

        from scvi.distributions import NegativeBinomial, Poisson, ZeroInflatedNegativeBinomial
        if cont_covs is None:
            decoder_input = z
        elif z.dim() != cont_covs.dim():
            decoder_input = torch.cat(
                [z, cont_covs.unsqueeze(0).expand(z.size(0), -1, -1)], dim=-1
            )
        else:
            decoder_input = torch.cat([z, cont_covs], dim=-1)

        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch

        if not self.use_size_factor_key:
            size_factor = library

        if self.batch_representation == "embedding":
            batch_rep = self.compute_embedding(REGISTRY_KEYS.BATCH_KEY, batch_index)
            decoder_input = torch.cat([decoder_input, batch_rep], dim=-1)
            px_scale, old_px_r, px_rate, px_dropout = self.decoder(
                self.dispersion,
                decoder_input,
                size_factor,
                *categorical_input,
                y,
            )
        else:
            px_scale, old_px_r, px_rate, px_dropout= self.decoder(
                self.dispersion,
                decoder_input,
                size_factor,
                batch_index,
                *categorical_input,
                y,
            )


        px_r = self.px_r_decoder(px_r)

        px_b = self.px_b
        #if self.dispersion == "gene-label":
         #   px_r = linear(
          #      one_hot(y.squeeze(-1), self.n_labels).float(), self.px_r
           # )  # px_r gets transposed - last dimension is nb genes
        #elif self.dispersion == "gene-batch":
         #   px_r = linear(one_hot(batch_index.squeeze(-1), self.n_batch).float(), self.px_r)
        #elif self.dispersion == "gene":
         #   px_r = self.px_r

        #px_r = torch.exp(px_r)

        if self.gene_likelihood == "zinb":
            px = ZeroInflatedNegativeBinomial(
                mu=px_rate,
                theta=px_r,
                zi_logits=px_dropout,
                scale=px_scale,
            )
        elif self.gene_likelihood == "nb":
            px = NegativeBinomial(mu=px_rate, theta=px_r, scale=px_scale)
        elif self.gene_likelihood == "poisson":
            px = Poisson(px_rate, scale=px_scale)
        elif self.gene_likelihood == "normal":
            px = Normal(px_rate, px_r)

        # Priors
        if self.use_observed_lib_size:
            pl = None
        else:
            (
                local_library_log_means,
                local_library_log_vars,
            ) = self._compute_local_library_params(batch_index)
            pl = Normal(local_library_log_means, local_library_log_vars.sqrt())
        pz = Normal(torch.zeros_like(z), torch.ones_like(z))


        if self.gbc_latent_dim is None:
            if self.b_prior_mixture:
                offset = (
                    10.0 * F.one_hot(y, num_classes=self.n_labels).float()
                    if self.n_labels >= 2
                    else 0.0
                )
                cats =torch.distributions.Categorical(logits=self.b_prior_logits + offset)
                normal_dists = torch.distributions.Normal(self.b_prior_means, torch.exp(self.b_prior_scales))
                pr = torch.distributions.MixtureSameFamily(cats, normal_dists)
            else:
                pr = Normal(
                    torch.zeros(self.b_dim, device=z.device),  # Mean 0
                    torch.ones(self.b_dim, device=z.device),    # Standard deviation 1
                )

        else:
            # Ensure guide_means and guide_vars are on the same device as z
            z_guide_m = self.z_guide_m.to(z.device)
            z_guide_v = self.z_guide_v.to(z.device)
            # Select the appropriate batch indices
            z_guide_m_batch = z_guide_m[batch_index]
            z_guide_v_batch = z_guide_v[batch_index]

            pr = Normal(
            z_guide_m_batch,  # Mean from guide_means
            torch.sqrt(z_guide_v_batch)  # Standard deviation from guide_vars
            )
            

        return {
            "px_b": px_b,
            "px_rate": px_rate,
            "px_r": px_r,
            "px_scale": px_scale,
            "pr": pr,
            MODULE_KEYS.PX_KEY: px,
            MODULE_KEYS.PL_KEY: pl,
            MODULE_KEYS.PZ_KEY: pz,
        }
    
    
    def loss(self, tensors, inference_outputs, generative_outputs, kl_weight: float = 1.0):
        from torch.distributions import kl_divergence

        x = tensors[REGISTRY_KEYS.X_KEY]
        
        n_obs_minibatch = x.shape[0] 
        kl_divergence_z = kl_divergence(inference_outputs[MODULE_KEYS.QZ_KEY], generative_outputs[MODULE_KEYS.PZ_KEY]).sum(dim=-1)
        if not self.use_observed_lib_size:
            kl_divergence_l = kl_divergence(
                inference_outputs[MODULE_KEYS.QL_KEY],
                generative_outputs[MODULE_KEYS.PL_KEY],
            ).sum(dim=1)
        else:
            kl_divergence_l = torch.tensor(0.0, device=x.device)

        px_rate = generative_outputs["px_rate"]
        px_r = generative_outputs["px_r"]
        px_b = generative_outputs["px_b"]

        reconst_loss = quasi_likelihood_loss(px_rate, x, px_r, px_b).sum(-1)

        
        if self.gbc_latent_dim is None:
            if self.b_prior_mixture:
                kl_b = inference_outputs[MODULE_KEYS.QR_KEY].log_prob(
                  inference_outputs[MODULE_KEYS.PX_R_KEY]
                ) - generative_outputs["pr"].log_prob(inference_outputs[MODULE_KEYS.PX_R_KEY])
                kl_b = kl_b.sum(-1)
            else:
                kl_b = kl_divergence(inference_outputs[MODULE_KEYS.QR_KEY], generative_outputs["pr"]).sum(-1)
        else:
            kl_b = kl_divergence(inference_outputs[MODULE_KEYS.QR_KEY], generative_outputs["pr"]).sum(-1)
    
    
        self.kl_r_log.append(kl_b.mean().item())


        kl_local_for_warmup = kl_divergence_z + kl_b
        kl_local_no_warmup = kl_divergence_l
        
        weighted_kl_local = kl_weight * kl_local_for_warmup + kl_local_no_warmup 

        loss = torch.mean(reconst_loss + weighted_kl_local)
        kl_local = {
            "kl_divergence_l": kl_divergence_l,
            "kl_divergence_z": kl_divergence_z,
            "kl_divergence_b": kl_b,  # kl divergence of b

        }
        return LossOutput(loss=loss, reconstruction_loss=reconst_loss, kl_local=kl_local, n_obs_minibatch=n_obs_minibatch)#, kl_divergence_b=kl_b.mean()) 
