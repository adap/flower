Disable/Enable Progress Bar
===========================

You will see a progress bar by default when you download a dataset or apply a map function. Here is how you control
this behavior.

Disable::

  from datasets.utils.logging import disable_progress_bar
  disable_progress_bar()

Enable::

  from datasets.utils.logging import enable_progress_bar
  enable_progress_bar()

