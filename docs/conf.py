project = "sajax"
author = "Samson Mercier, Lucas Arthur"

extensions = [
    "myst_nb",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "autoapi.extension",
    "sphinx.ext.doctest",
    "matplotlib.sphinxext.plot_directive",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


html_theme = "sphinx_book_theme"
html_static_path = ["_static"]

source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".myst": "myst-nb",
}

root_doc = "index"

html_theme_options = {
    "repository_url": "https://github.com/SamMerc/sajax",
    "use_repository_button": True,
    "path_to_docs": "docs",
    "repository_branch": "main",
    "launch_buttons": {
        # Binder builds the environment from binder/requirements.txt (an
        # `uv export` of the docs+examples dependency set -- regenerate with
        # `uv export --format requirements-txt --extra docs --group examples
        # --no-hashes --quiet > binder/requirements.txt` after touching
        # pyproject.toml/uv.lock) and binder/runtime.txt (Python version).
        # Colab has no repo checkout, so each notebook installs itself from
        # that same file in its own first cell -- see there for details.
        "binderhub_url": "https://mybinder.org",
        "colab_url": "https://colab.research.google.com",
        "notebook_interface": "jupyterlab",
    },
}

nb_render_image_options = {"align": "center"}

myst_enable_extensions = [
    "dollarmath",
]

html_logo = "_static/logo.png"
myst_url_schemes = ("http", "https")

plot_html_show_formats = False
plot_html_show_source_link = False

autoapi_dirs = ["../sajax"]
autoapi_ignore = ["*_version*", "*/types*"]
autoapi_options = [
    "members",
    "undoc-members",
    # "private-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
    # "imported-members",
]
# autoapi_add_toctree_entry = False
autoapi_template_dir = "_autoapi_templates"

suppress_warnings = ["autoapi.python_import_resolution"]

nb_execution_excludepatterns = []
plot_include_source = True