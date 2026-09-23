"""Transparent operator call accounting for reproducible computation budgets."""


class CountedOperator:
    def __init__(self, operator):
        self.operator = operator
        self.counts = dict(likelihood_gradients=0, amplitude_projections=0, direct_forward_calls=0)

    def __getattr__(self, name):
        return getattr(self.operator, name)

    def gradient(self, *args, **kwargs):
        self.counts['likelihood_gradients'] += 1
        return self.operator.gradient(*args, **kwargs)

    def project_amplitude(self, *args, **kwargs):
        self.counts['amplitude_projections'] += 1
        return self.operator.project_amplitude(*args, **kwargs)

    def forward(self, *args, **kwargs):
        self.counts['direct_forward_calls'] += 1
        return self.operator.forward(*args, **kwargs)

    def vjp(self, *args, **kwargs):
        self.counts['initialization_vjps'] = self.counts.get('initialization_vjps', 0) + 1
        return self.operator.vjp(*args, **kwargs)
