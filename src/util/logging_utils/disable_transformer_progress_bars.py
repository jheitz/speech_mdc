def disable_transformer_progress_bars():
    # prevent tqdm progress bars from transformers, as they bloat the error logs
    from transformers.utils import logging
    logging.disable_progress_bar()
