// Paper links identify the method or explicitly named baseline; source links pin the benchmark revision.
window.METHOD_DETAILS = {
  "PermutationSamplingSII": {
    "description": "Samples random player orderings and averages joint marginal contributions to estimate SII or k-SII interactions. The linked SHAP-IQ paper describes this permutation baseline in Appendix D.1.",
    "paper": {
      "title": "SHAP-IQ: Unified Approximation of any-order Shapley Interactions",
      "url": "https://arxiv.org/abs/2303.01179"
    },
    "implementation": {
      "title": "shapiq implementation",
      "url": "https://github.com/rtealwitter/shapiq/blob/1472a0035f4e54df9eb4bdf6f771330f8b93d856/src/shapiq/approximator/permutation/sii.py#L21"
    }
  },
  "PermutationSamplingSTII": {
    "description": "Uses random player orderings to estimate Shapley-Taylor interactions, with lower orders computed directly. This targets STII, whose interaction definition differs from SII.",
    "paper": {
      "title": "The Shapley Taylor Interaction Index",
      "url": "https://arxiv.org/abs/1902.05622"
    },
    "implementation": {
      "title": "shapiq implementation",
      "url": "https://github.com/rtealwitter/shapiq/blob/1472a0035f4e54df9eb4bdf6f771330f8b93d856/src/shapiq/approximator/permutation/stii.py#L23"
    }
  },
  "PermutationSamplingSV": {
    "description": "Averages each player’s marginal contribution as players join in random order. Each sampled ordering supplies contributions for every player; this variant estimates Shapley values.",
    "paper": {
      "title": "Polynomial calculation of the Shapley value based on sampling",
      "url": "https://doi.org/10.1016/j.cor.2008.04.004"
    },
    "implementation": {
      "title": "shapiq implementation",
      "url": "https://github.com/rtealwitter/shapiq/blob/1472a0035f4e54df9eb4bdf6f771330f8b93d856/src/shapiq/approximator/permutation/sv.py#L22"
    }
  },
  "StratifiedSamplingSV": {
    "description": "Groups marginal-contribution samples by coalition size and combines the group averages. This balances the different coalition sizes when estimating Shapley values.",
    "paper": {
      "title": "Bounding the Estimation Error of Sampling-based Shapley Value Approximation",
      "url": "https://arxiv.org/abs/1306.4265"
    },
    "implementation": {
      "title": "shapiq implementation",
      "url": "https://github.com/rtealwitter/shapiq/blob/1472a0035f4e54df9eb4bdf6f771330f8b93d856/src/shapiq/approximator/marginals/stratified.py#L21"
    }
  },
  "OwenSamplingSV": {
    "description": "Estimates Shapley values through their multilinear integral representation. It samples marginal contributions at several player-inclusion probabilities and averages across them.",
    "paper": {
      "title": "A Multilinear Sampling Algorithm to Estimate Shapley Values",
      "url": "https://arxiv.org/abs/2010.12082"
    },
    "implementation": {
      "title": "shapiq implementation",
      "url": "https://github.com/rtealwitter/shapiq/blob/1472a0035f4e54df9eb4bdf6f771330f8b93d856/src/shapiq/approximator/marginals/owen.py#L21"
    }
  },
  "KernelSHAP": {
    "description": "Fits a weighted linear model to sampled coalition values using the Shapley kernel. Its coefficients estimate individual Shapley values.",
    "paper": {
      "title": "A Unified Approach to Interpreting Model Predictions",
      "url": "https://arxiv.org/abs/1705.07874"
    },
    "implementation": {
      "title": "shapiq implementation",
      "url": "https://github.com/rtealwitter/shapiq/blob/1472a0035f4e54df9eb4bdf6f771330f8b93d856/src/shapiq/approximator/regression/kernelshap.py#L17"
    }
  },
  "LeverageSHAP": {
    "description": "Estimates Shapley values using leverage-score sampling and efficiency-constrained regression. Budgets at most 3× players use a 0.001 ridge safeguard, except when all coalitions are evaluated.",
    "paper": {
      "title": "Provably Accurate Shapley Value Estimation via Leverage Score Sampling",
      "url": "https://arxiv.org/abs/2410.01917"
    },
    "implementation": {
      "title": "shapiq implementation",
      "url": "https://github.com/rtealwitter/shapiq/blob/d4ac18e674841f79c1ca25d8cfbf550e84dc21a7/src/shapiq/approximator/regression/leverageshap.py#L26"
    }
  },
  "RegressionFSII": {
    "description": "Fits a weighted polynomial to coalition values to estimate Faithful Shapley interactions (FSII). At order one, its target is the ordinary Shapley value.",
    "paper": {
      "title": "Faith-Shap: The Faithful Shapley Interaction Index",
      "url": "https://arxiv.org/abs/2203.00870"
    },
    "implementation": {
      "title": "shapiq implementation",
      "url": "https://github.com/rtealwitter/shapiq/blob/1472a0035f4e54df9eb4bdf6f771330f8b93d856/src/shapiq/approximator/regression/faithful.py#L19"
    }
  },
  "RegressionFBII": {
    "description": "Fits a polynomial under uniform coalition weighting to estimate Faithful Banzhaf interactions (FBII). These are Banzhaf-based interactions; the order-one target is the Banzhaf value.",
    "paper": {
      "title": "Faith-Shap: The Faithful Shapley Interaction Index",
      "url": "https://arxiv.org/abs/2203.00870"
    },
    "implementation": {
      "title": "shapiq implementation",
      "url": "https://github.com/rtealwitter/shapiq/blob/1472a0035f4e54df9eb4bdf6f771330f8b93d856/src/shapiq/approximator/regression/faithful.py#L77"
    }
  },
  "KernelSHAPIQ": {
    "description": "Extends KernelSHAP to SII and k-SII interactions through weighted regressions applied successively by interaction order. Its order-one specialization estimates Shapley values.",
    "paper": {
      "title": "KernelSHAP-IQ: Weighted Least-Square Optimization for Shapley Interactions",
      "url": "https://arxiv.org/abs/2405.10852"
    },
    "implementation": {
      "title": "shapiq implementation",
      "url": "https://github.com/rtealwitter/shapiq/blob/1472a0035f4e54df9eb4bdf6f771330f8b93d856/src/shapiq/approximator/regression/kernelshapiq.py#L16"
    }
  },
  "InconsistentKernelSHAPIQ": {
    "description": "A related regression variant for SII and k-SII, retained as a comparison method. At higher orders, its estimates need not converge to the true SII even with more samples.",
    "paper": {
      "title": "KernelSHAP-IQ: Weighted Least-Square Optimization for Shapley Interactions",
      "url": "https://arxiv.org/abs/2405.10852"
    },
    "implementation": {
      "title": "shapiq implementation",
      "url": "https://github.com/rtealwitter/shapiq/blob/1472a0035f4e54df9eb4bdf6f771330f8b93d856/src/shapiq/approximator/regression/kernelshapiq.py#L87"
    }
  },
  "ProxySPEX": {
    "description": "Fits a tree surrogate to sampled coalition values, extracts its Fourier interactions, then refines their coefficients against the sampled game. The resulting sparse representation is converted to the requested value or interaction index.",
    "paper": {
      "title": "ProxySPEX: Inference-Efficient Interpretability via Sparse Feature Interactions in LLMs",
      "url": "https://arxiv.org/abs/2505.17495"
    },
    "implementation": {
      "title": "shapiq implementation",
      "url": "https://github.com/rtealwitter/shapiq/blob/1472a0035f4e54df9eb4bdf6f771330f8b93d856/src/shapiq/approximator/proxy/proxyspex.py#L36"
    }
  },
  "ProxySHAP": {
    "description": "Fits a regression surrogate to sampled coalition values and reads values or interactions from that surrogate. An optional Monte Carlo residual correction addresses surrogate error; it is disabled by default in this benchmark.",
    "paper": {
      "title": "Proxy-Based Approximation of Shapley and Banzhaf Interactions",
      "url": "https://arxiv.org/abs/2605.22738"
    },
    "implementation": {
      "title": "shapiq implementation",
      "url": "https://github.com/rtealwitter/shapiq/blob/1472a0035f4e54df9eb4bdf6f771330f8b93d856/src/shapiq/approximator/proxy/proxyshap.py#L160"
    }
  },
  "OddSHAP": {
    "description": "Uses a tree surrogate to select odd-order Fourier terms, then fits those terms to estimate Shapley values. Its budget-dependent support selection can omit individual players at small budgets.",
    "paper": {
      "title": "An Odd Estimator for Shapley Values",
      "url": "https://arxiv.org/abs/2602.01399"
    },
    "implementation": {
      "title": "shapiq implementation",
      "url": "https://github.com/rtealwitter/shapiq/blob/1472a0035f4e54df9eb4bdf6f771330f8b93d856/src/shapiq/approximator/regression/oddshap.py#L109"
    }
  },
  "RegressionMSR": {
    "description": "Combines exact attributions of a fitted surrogate with a Monte Carlo correction for the surrogate’s residual error. Unlike the benchmark’s ProxySHAP default, it enables residual adjustment and uses a different sampling distribution. This implementation supports individual Shapley and Banzhaf values.",
    "paper": {
      "title": "Regression-adjusted Monte Carlo Estimators for Shapley Values and Probabilistic Values",
      "url": "https://arxiv.org/abs/2506.11849"
    },
    "implementation": {
      "title": "shapiq implementation",
      "url": "https://github.com/rtealwitter/shapiq/blob/1472a0035f4e54df9eb4bdf6f771330f8b93d856/src/shapiq/approximator/proxy/regressionmsr.py#L38"
    }
  },
  "ShaplEIG": {
    "description": "Fits a Gaussian-process surrogate and chooses additional coalitions by their expected information gain about Shapley values. This prioritizes sample efficiency and can require substantial computation per query.",
    "paper": {
      "title": "ShaplEIG: Bayesian Experimental Design for Shapley Value Estimation",
      "url": "https://arxiv.org/abs/2606.02247"
    },
    "implementation": {
      "title": "shapiq implementation",
      "url": "https://github.com/rtealwitter/shapiq/blob/1472a0035f4e54df9eb4bdf6f771330f8b93d856/src/shapiq/approximator/shapleig/shapleig.py#L35"
    }
  },
  "SHAPIQ": {
    "description": "Reuses each sampled coalition value across the requested interaction scores through a Monte Carlo representation. It supports Shapley values and several higher-order interaction definitions.",
    "paper": {
      "title": "SHAP-IQ: Unified Approximation of any-order Shapley Interactions",
      "url": "https://arxiv.org/abs/2303.01179"
    },
    "implementation": {
      "title": "shapiq implementation",
      "url": "https://github.com/rtealwitter/shapiq/blob/1472a0035f4e54df9eb4bdf6f771330f8b93d856/src/shapiq/approximator/montecarlo/shapiq.py#L22"
    }
  },
  "SVARM": {
    "description": "Reuses coalition evaluations across players and stratifies contributions by coalition size and player membership. This variant estimates individual Shapley or Banzhaf values.",
    "paper": {
      "title": "Approximating the Shapley Value without Marginal Contributions",
      "url": "https://arxiv.org/abs/2302.00736"
    },
    "implementation": {
      "title": "shapiq implementation",
      "url": "https://github.com/rtealwitter/shapiq/blob/1472a0035f4e54df9eb4bdf6f771330f8b93d856/src/shapiq/approximator/montecarlo/svarmiq.py#L75"
    }
  },
  "SVARMIQ": {
    "description": "Extends stratified coalition reuse to higher-order interactions, grouping samples by coalition size and overlap with each interaction. It supports several interaction definitions as well as individual values.",
    "paper": {
      "title": "SVARM-IQ: Efficient Approximation of Any-order Shapley Interactions through Stratification",
      "url": "https://arxiv.org/abs/2401.13371"
    },
    "implementation": {
      "title": "shapiq implementation",
      "url": "https://github.com/rtealwitter/shapiq/blob/1472a0035f4e54df9eb4bdf6f771330f8b93d856/src/shapiq/approximator/montecarlo/svarmiq.py#L16"
    }
  },
  "kADDSHAP": {
    "description": "Approximates the game using a k-additive representation fitted by regression. This benchmark uses its order-one Shapley-value configuration; higher-order kADD coefficients define a different target.",
    "paper": {
      "title": "A k-additive Choquet integral-based approach to approximate the SHAP values for local interpretability in machine learning",
      "url": "https://arxiv.org/abs/2211.02166"
    },
    "implementation": {
      "title": "shapiq implementation",
      "url": "https://github.com/rtealwitter/shapiq/blob/1472a0035f4e54df9eb4bdf6f771330f8b93d856/src/shapiq/approximator/regression/kadd_shap.py#L15"
    }
  },
  "SPEX": {
    "description": "Recovers a sparse Fourier representation of the game using structured coalition queries, then converts it to values or interactions. Its transform requires a minimum number of queries before it can return an estimate.",
    "paper": {
      "title": "SPEX: Scaling Feature Interaction Explanations for LLMs",
      "url": "https://arxiv.org/abs/2502.13870"
    },
    "implementation": {
      "title": "shapiq implementation",
      "url": "https://github.com/rtealwitter/shapiq/blob/1472a0035f4e54df9eb4bdf6f771330f8b93d856/src/shapiq/approximator/sparse/spex.py#L10"
    }
  },
  "UnbiasedKernelSHAP": {
    "description": "Uses the unbiased Shapley-value estimator derived from KernelSHAP’s regression formulation. In shapiq it is implemented as the order-one specialization of SHAP-IQ.",
    "paper": {
      "title": "Improving KernelSHAP: Practical Shapley Value Estimation via Linear Regression",
      "url": "https://arxiv.org/abs/2012.01536"
    },
    "implementation": {
      "title": "shapiq implementation",
      "url": "https://github.com/rtealwitter/shapiq/blob/1472a0035f4e54df9eb4bdf6f771330f8b93d856/src/shapiq/approximator/montecarlo/shapiq.py#L100"
    }
  }
};
