{{ fullname | escape | underline }}

.. automodule:: {{ fullname }}
   :no-members:

{% if classes %}
.. rubric:: Classes

.. autosummary::
   :toctree:
   :nosignatures:
{% for item in classes %}
   {{ item }}
{%- endfor %}
{% endif %}

{% if functions %}
.. rubric:: Functions

.. autosummary::
   :toctree:
   :nosignatures:
{% for item in functions %}
   {{ item }}
{%- endfor %}
{% endif %}

{% if attributes %}
.. rubric:: Module Attributes

.. autosummary::
   :toctree:
   :nosignatures:
{% for item in attributes %}
   {{ item }}
{%- endfor %}
{% endif %}
