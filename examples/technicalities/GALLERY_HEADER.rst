Technicalities
==============

.. container:: shapiq-gallery-hero

   .. image:: /_static/gallery/technicalities.svg
      :alt: A masked feature vector imputed from background data
      :class: shapiq-gallery-hero-image

   .. container:: shapiq-gallery-hero-text

      How ``shapiq`` turns a model into a game, and how to explain many points fast. To evaluate a
      coalition, an imputer fills in the features outside it: the marginal and baseline imputers
      take values from background data or a fixed baseline (interventional), and the conditional
      imputers sample values that fit the present features (observational). The choice changes
      what an explanation means.

      Explanations of many points can run in parallel with ``n_jobs``.

.. rst-class:: shapiq-gallery-api

**Key API:** :class:`~shapiq.imputer.MarginalImputer` · :class:`~shapiq.imputer.BaselineImputer` ·
:class:`~shapiq.imputer.GenerativeConditionalImputer` · :class:`~shapiq.imputer.GaussianImputer` ·
:class:`~shapiq.imputer.GaussianCopulaImputer` · :meth:`~shapiq.explainer.Explainer.explain_X`
