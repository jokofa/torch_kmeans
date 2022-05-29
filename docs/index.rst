============
torch_kmeans
============

This is the documentation of **torch_kmeans**.

============
torch_kmeans
============

   PyTorch implementations of KMeans, Soft-KMeans and Constrained-KMeans


torch_kmeans features implementations of the well known k-means algorithm
as well as its soft and constrained variants.

All algorithms are completely implemented as `PyTorch <https://pytorch.org/>`_ modules
and can be easily incorporated in a PyTorch pipeline or model.
Therefore, they support execution on GPU as well as working on (mini-)batches of data.
Moreover, they also provide a `scikit-learn <https://scikit-learn.org/>`_ style interface featuring

.. code-block:: python

   model.fit(), model.predict() and model.fit_predict()


functions.

-> `view official github <https://github.com/jokofa/torch_kmeans>`_


Highlights
===========
- Fully implemented in PyTorch.
- GPU support like native PyTorch.
- PyTorch script JIT compiled for most performance sensitive parts.
- Works with mini-batches of samples:
   - each instance can have a different number of clusters.
- Constrained Kmeans works with cluster constraints like:
   - a max number of samples per cluster or,
   - a maximum weight per cluster, where each sample has an associated weight.
- SoftKMeans is a fully differentiable clustering procedure and
  can readily be used in a PyTorch neural network model which requires backpropagation.
- Unit tested against the scikit-learn KMeans implementation.
- GPU execution enables very fast computation even for
  large batch size or very high dimensional feature spaces
  (see `speed comparison <https://github.com/jokofa/torch_kmeans/tree/master/examples/notebooks/speed_comparison.ipynb>`_)

Installation
=============

Simply install from PyPI

.. code-block:: console

   pip install torch-kmeans


Contents
========

..    .. toctree::
       :maxdepth: 2
       Contributions & Help <contributing>
       Changelog <changelog>


.. toctree::
   :maxdepth: 2

   Overview <readme>
   License <license>
   Authors <authors>
   Module Reference <api/modules>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _toctree: https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html
.. _reStructuredText: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
.. _references: https://www.sphinx-doc.org/en/stable/markup/inline.html
.. _Python domain syntax: https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#the-python-domain
.. _Sphinx: https://www.sphinx-doc.org/
.. _Python: https://docs.python.org/
.. _Numpy: https://numpy.org/doc/stable
.. _SciPy: https://docs.scipy.org/doc/scipy/reference/
.. _matplotlib: https://matplotlib.org/contents.html#
.. _Pandas: https://pandas.pydata.org/pandas-docs/stable
.. _Scikit-Learn: https://scikit-learn.org/stable
.. _autodoc: https://www.sphinx-doc.org/en/master/ext/autodoc.html
.. _Google style: https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
.. _NumPy style: https://numpydoc.readthedocs.io/en/latest/format.html
.. _classical style: https://www.sphinx-doc.org/en/master/domains.html#info-field-lists
