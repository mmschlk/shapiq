{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:
   :special-members: __init__

   {% block methods %}
   {% if methods %}
   .. rubric:: Methods

   .. autosummary::
      :toctree:
      :nosignatures:
   {% for item in methods %}
      ~{{ fullname }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: Attributes

   .. autosummary::
      :toctree:
      :nosignatures:
   {% for item in attributes %}
      ~{{ fullname }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
