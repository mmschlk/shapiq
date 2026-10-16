{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:
{#- the games and setups inherit much of their API from private base classes: show it, but not
    the members of shapiq.Game or of the standard library's bases #}
{%- if module.split(".")[0] in ["shapiq_games", "shapiq_benchmark"] %}
   :inherited-members: Game, ABC, Generic, BaseException
{%- endif %}
