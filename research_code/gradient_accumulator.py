import tensorflow as tf


class GradientAccumulator(object):
    """
    Distribution strategies-aware gradient accumulation utility.
    from https://huggingface.co/transformers/main_classes/optimizer_schedules.html#transformers.GradientAccumulator
    """

    def __init__(self):
        """Initializes the accumulator."""
        self._gradients = []
        self._accum_steps = tf.Variable(
            initial_value=0, dtype=tf.int64, trainable=False, aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA
        )

    @property
    def step(self):
        """Number of accumulated steps."""
        return self._accum_steps.value()

    @property
    def gradients(self):
        """The accumulated gradients."""
        return list(
            gradient.value() if gradient is not None else gradient for gradient in self._get_replica_gradients()
        )

    def __call__(self, gradients):
        """Accumulates :obj:`gradients`."""
        if not self._gradients:
            self._gradients.extend(
                [
                    tf.Variable(tf.zeros_like(gradient), trainable=False) if gradient is not None else gradient
                    for gradient in gradients
                ]
            )

        if len(gradients) != len(self._gradients):
            raise ValueError("Expected %s gradients, but got %d" % (len(self._gradients), len(gradients)))

        for accum_gradient, gradient in zip(self._get_replica_gradients(), gradients):
            if accum_gradient is not None and gradient is not None:
                accum_gradient.assign_add(gradient)

        self._accum_steps.assign_add(1)

    def reset(self):
        """Resets the accumulated gradients."""
        if self._gradients:
            self._accum_steps.assign(0)

        for gradient in self._get_replica_gradients():
            if gradient is not None:
                gradient.assign(tf.zeros_like(gradient))

    def _get_replica_gradients(self):
        if tf.distribute.has_strategy():
            # In a replica context, we want to accumulate gradients on each replica
            # without synchronization, so we directly assign the value of the
            # current replica.
            replica_context = tf.distribute.get_replica_context()

            if replica_context is None or tf.distribute.get_strategy().num_replicas_in_sync == 1:
                return self._gradients

            return (
                gradient.device_map.select_for_current_replica(gradient.values, replica_context)
                for gradient in self._gradients
                if gradient is not None
            )
        else:
            return self._gradients