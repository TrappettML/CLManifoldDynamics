class CLHook:
    """
    Base class for hooks. 
    Override these methods to inject logic into the training loop.
    """
    def on_task_start(self, task, state):
        """Called at the beginning of a new task."""
        pass

    def on_task_end(self, task, state, metrics):
        """Called at the end of a task."""
        pass

    def on_epoch_start(self, epoch, state):
        """Called at the start of an epoch."""
        pass

    def on_epoch_end(self, epoch, state, metrics):
        """Called at the end of an epoch."""
        pass
    
    # Note: adding on_batch_start/end inside JAX scans is complex 
    # due to pure function constraints, but we can return auxiliary data 
    # from the train step if needed.