from itertools import izip

import numpy as np
import numpy.random as rng
import theano
import theano.tensor as tt

import layers as layers
import mades as mades
import util

dtype = theano.config.floatX


class MaskedAutoregressiveFlow:
    """
    Implements a Masked Autoregressive Flow, which is a stack of mades such that the random numbers which drive made i
    are generated by made i-1. The first made is driven by standard gaussian noise. In the current implementation, all
    mades are of the same type. If there is only one made in the stack, then it's equivalent to a single made.
    """

    def __init__(self, n_inputs, n_hiddens, act_fun, n_mades, batch_norm=True, input_order='sequential', mode='sequential', input=None):
        """
        Constructor.
        :param n_inputs: number of inputs
        :param n_hiddens: list with number of hidden units for each hidden layer
        :param act_fun: name of activation function
        :param n_mades: number of mades
        :param batch_norm: whether to use batch normalization between mades
        :param input_order: order of inputs of last made
        :param mode: strategy for assigning degrees to hidden nodes: can be 'random' or 'sequential'
        :param input: theano variable to serve as input; if None, a new variable is created
        """

        # save input arguments
        self.n_inputs = n_inputs
        self.n_hiddens = n_hiddens
        self.act_fun = act_fun
        self.n_mades = n_mades
        self.batch_norm = batch_norm
        self.mode = mode

        self.input = tt.matrix('x', dtype=dtype) if input is None else input
        self.parms = []

        self.mades = []
        self.bns = []
        self.u = self.input
        self.logdet_dudx = 0.0

        for i in xrange(n_mades):

            # create a new made
            made = mades.GaussianMade(n_inputs, n_hiddens, act_fun, input_order, mode, self.u)
            self.mades.append(made)
            self.parms += made.parms
            input_order = input_order if input_order == 'random' else made.input_order[::-1]

            # inverse autoregressive transform
            self.u = made.u
            self.logdet_dudx += 0.5 * tt.sum(made.logp, axis=1)

            # batch normalization
            if batch_norm:
                bn = layers.BatchNorm(self.u, n_inputs)
                self.u = bn.y
                self.parms += bn.parms
                self.logdet_dudx += tt.sum(bn.log_gamma) - 0.5 * tt.sum(tt.log(bn.v))
                self.bns.append(bn)

        self.input_order = self.mades[0].input_order

        # log likelihoods
        self.L = -0.5 * n_inputs * np.log(2 * np.pi) - 0.5 * tt.sum(self.u ** 2, axis=1) + self.logdet_dudx
        self.L.name = 'L'

        # train objective
        self.trn_loss = -tt.mean(self.L)
        self.trn_loss.name = 'trn_loss'

        # theano evaluation functions, will be compiled when first needed
        self.eval_lprob_f = None
        self.eval_us_f = None
        self.eval_grad_f = None

    def eval(self, x, log=True):
        """
        Evaluate log probabilities for given inputs.
        :param x: data matrix where rows are inputs
        :param log: whether to return probabilities in the log domain
        :return: list of log probabilities log p(x)
        """

        # compile theano function, if haven't already done so
        if self.eval_lprob_f is None:
            self.eval_lprob_f = theano.function(
                inputs=[self.input],
                outputs=self.L,
                givens=[(bn.m, bn.bm) for bn in self.bns] + [(bn.v, bn.bv) for bn in self.bns]
            )

        lprob = self.eval_lprob_f(x.astype(dtype))

        return lprob if log else np.exp(lprob)

    def grad(self, x):
        if self.eval_grad_f is None:
            const = []
            val = []
            for bn in self.bns:
                const.extend([bn.m, bn.v])
                val.extend([(bn.m, bn.bm), (bn.v, bn.bv)])

            self.eval_grad_f = theano.function(
                inputs=[self.input],
                outputs=tt.grad(self.L[0], self.input, consider_constant=const),
                givens=val)

        return util.grad_helper(x.astype(dtype), self.eval_grad_f, self.eval)

    def gen(self, n_samples=1, u=None):
        """
        Generate samples, by propagating random numbers through each made.
        :param n_samples: number of samples
        :param u: random numbers to use in generating samples; if None, new random numbers are drawn
        :return: samples
        """

        x = rng.randn(n_samples, self.n_inputs).astype(dtype) if u is None else u

        if getattr(self, 'batch_norm', False):

            for made, bn in izip(self.mades[::-1], self.bns[::-1]):
                x = bn.eval_inv(x)
                x = made.gen(n_samples, x)

        else:

            for made in self.mades[::-1]:
                x = made.gen(n_samples, x)

        return x

    def calc_random_numbers(self, x):
        """
        Givan a dataset, calculate the random numbers used internally to generate the dataset.
        :param x: numpy array, rows are datapoints
        :return: numpy array, rows are corresponding random numbers
        """

        # compile theano function, if haven't already done so
        if self.eval_us_f is None:
            self.eval_us_f = theano.function(
                inputs=[self.input],
                outputs=self.u
            )

        return self.eval_us_f(x.astype(dtype))


class ConditionalMaskedAutoregressiveFlow:
    """
    Implements a Conditional Masked Autoregressive Flow.
    """

    def __init__(self, n_inputs, n_outputs, n_hiddens, act_fun, n_mades, batch_norm=True, output_order='sequential', mode='sequential', input=None, output=None):
        """
        Constructor.
        :param n_inputs: number of (conditional) inputs
        :param n_outputs: number of outputs
        :param n_hiddens: list with number of hidden units for each hidden layer
        :param act_fun: name of activation function
        :param n_mades: number of mades in the flow
        :param batch_norm: whether to use batch normalization between mades in the flow
        :param output_order: order of outputs of last made
        :param mode: strategy for assigning degrees to hidden nodes: can be 'random' or 'sequential'
        :param input: theano variable to serve as input; if None, a new variable is created
        :param output: theano variable to serve as output; if None, a new variable is created
        """

        # save input arguments
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hiddens = n_hiddens
        self.act_fun = act_fun
        self.n_mades = n_mades
        self.batch_norm = batch_norm
        self.mode = mode

        self.input = tt.matrix('x', dtype=dtype) if input is None else input
        self.y = tt.matrix('y', dtype=dtype) if output is None else output
        self.parms = []

        self.mades = []
        self.bns = []
        self.u = self.y
        self.logdet_dudy = 0.0

        for i in xrange(n_mades):

            # create a new made
            made = mades.ConditionalGaussianMade(n_inputs, n_outputs, n_hiddens, act_fun, output_order, mode, self.input, self.u)
            self.mades.append(made)
            self.parms += made.parms
            output_order = output_order if output_order == 'random' else made.output_order[::-1]

            # inverse autoregressive transform
            self.u = made.u
            self.logdet_dudy += 0.5 * tt.sum(made.logp, axis=1)

            # batch normalization
            if batch_norm:
                bn = layers.BatchNorm(self.u, n_outputs)
                self.u = bn.y
                self.parms += bn.parms
                self.logdet_dudy += tt.sum(bn.log_gamma) - 0.5 * tt.sum(tt.log(bn.v))
                self.bns.append(bn)

        self.output_order = self.mades[0].output_order

        # log likelihoods
        self.L = -0.5 * n_outputs * np.log(2 * np.pi) - 0.5 * tt.sum(self.u ** 2, axis=1) + self.logdet_dudy
        self.L.name = 'L'

        # train objective
        self.trn_loss = -tt.mean(self.L)
        self.trn_loss.name = 'trn_loss'

        # theano evaluation functions, will be compiled when first needed
        self.eval_lprob_f = None
        self.eval_us_f = None

    def eval(self, xy, log=True):
        """
        Evaluate log probabilities for given input-output pairs.
        :param xy: a pair (x, y) where x rows are inputs and y rows are outputs
        :param log: whether to return probabilities in the log domain
        :return: log probabilities: log p(y|x)
        """

        # compile theano function, if haven't already done so
        if self.eval_lprob_f is None:
            self.eval_lprob_f = theano.function(
                inputs=[self.input, self.y],
                outputs=self.L,
                givens=[(bn.m, bn.bm) for bn in self.bns] + [(bn.v, bn.bv) for bn in self.bns]
            )

        x, y = xy
        lprob = self.eval_lprob_f(x.astype(dtype), y.astype(dtype))

        return lprob if log else np.exp(lprob)

    def gen(self, x, n_samples=1, u=None):
        """
        Generate samples, by propagating random numbers through each made, after conditioning on input x.
        :param x: input vector
        :param n_samples: number of samples
        :param u: random numbers to use in generating samples; if None, new random numbers are drawn
        :return: samples
        """

        y = rng.randn(n_samples, self.n_outputs).astype(dtype) if u is None else u

        if getattr(self, 'batch_norm', False):

            for made, bn in izip(self.mades[::-1], self.bns[::-1]):
                y = bn.eval_inv(y)
                y = made.gen(x, n_samples, y)

        else:

            for made in self.mades[::-1]:
                y = made.gen(x, n_samples, y)

        return y

    def calc_random_numbers(self, xy):
        """
        Givan a dataset, calculate the random numbers used internally to generate the dataset.
        :param xy: a pair (x, y) of numpy arrays, where x rows are inputs and y rows are outputs
        :return: numpy array, rows are corresponding random numbers
        """

        # compile theano function, if haven't already done so
        if self.eval_us_f is None:
            self.eval_us_f = theano.function(
                inputs=[self.input, self.y],
                outputs=self.u
            )

        x, y = xy
        return self.eval_us_f(x.astype(dtype), y.astype(dtype))


class MaskedAutoregressiveFlow_on_MADE:
    """
    A Masked Autoregressive Flow, where the target distribution is given by a MoG MADE.
    """

    def __init__(self, n_inputs, n_hiddens, act_fun, n_layers, n_comps, batch_norm=True, input_order='sequential', mode='sequential', input=None):
        """
        Constructor.
        :param n_inputs: number of inputs
        :param n_hiddens: list with number of hidden units for each hidden layer
        :param act_fun: name of activation function
        :param n_layers: number of layers in the flow
        :param n_comps: number of gaussians per conditional for the target made
        :param batch_norm: whether to use batch normalization between layers
        :param input_order: order of inputs of last made
        :param mode: strategy for assigning degrees to hidden nodes: can be 'random' or 'sequential'
        :param input: theano variable to serve as input; if None, a new variable is created
        """

        # save input arguments
        self.n_inputs = n_inputs
        self.n_hiddens = n_hiddens
        self.act_fun = act_fun
        self.n_layers = n_layers
        self.n_comps = n_comps
        self.batch_norm = batch_norm
        self.mode = mode

        self.input = tt.matrix('x', dtype=dtype) if input is None else input
        self.parms = []

        # maf
        self.maf = MaskedAutoregressiveFlow(n_inputs, n_hiddens, act_fun, n_layers, batch_norm, input_order, mode, self.input)
        self.bns = self.maf.bns
        self.parms += self.maf.parms
        self.input_order = self.maf.input_order

        # mog made
        input_order = input_order if input_order == 'random' else self.maf.mades[-1].input_order[::-1]
        self.made = mades.MixtureOfGaussiansMade(n_inputs, n_hiddens, act_fun, n_comps, input_order, mode, self.maf.u)
        self.parms += self.made.parms

        # log likelihoods
        self.L = self.made.L + self.maf.logdet_dudx
        self.L.name = 'L'

        # train objective
        self.trn_loss = -tt.mean(self.L)
        self.trn_loss.name = 'trn_loss'

        # theano evaluation functions, will be compiled when first needed
        self.eval_lprob_f = None
        self.eval_us_f = None
        self.eval_grad_f = None

    def eval(self, x, log=True):
        """
        Evaluate log probabilities for given inputs.
        :param x: data matrix where rows are inputs
        :param log: whether to return probabilities in the log domain
        :return: list of log probabilities log p(x)
        """

        # compile theano function, if haven't already done so
        if self.eval_lprob_f is None:
            self.eval_lprob_f = theano.function(
                inputs=[self.input],
                outputs=self.L,
                givens=[(bn.m, bn.bm) for bn in self.bns] + [(bn.v, bn.bv) for bn in self.bns]
            )

        lprob = self.eval_lprob_f(x.astype(dtype))

        return lprob if log else np.exp(lprob)

    def grad(self, x):
        if self.eval_grad_f is None:
            const = []
            val = []
            for bn in self.bns:
                const.extend([bn.m, bn.v])
                val.extend([(bn.m, bn.bm), (bn.v, bn.bv)])

            self.eval_grad_f = theano.function(
                inputs=[self.input],
                outputs=tt.grad(self.L[0], self.input, consider_constant=const),
                givens=val)

        return util.grad_helper(x.astype(dtype), self.eval_grad_f, self.eval)

    def gen(self, n_samples=1, u=None):
        """
        Generate samples, by propagating random numbers through each made.
        :param n_samples: number of samples
        :param u: random numbers to use in generating samples; if None, new random numbers are drawn
        :return: samples
        """

        x = rng.randn(n_samples, self.n_inputs).astype(dtype) if u is None else u
        x = self.made.gen(n_samples, x)
        x = self.maf.gen(n_samples, x)

        return x

    def calc_random_numbers(self, x):
        """
        Givan a dataset, calculate the random numbers used internally to generate the dataset.
        :param x: numpy array, rows are datapoints
        :return: numpy array, rows are corresponding random numbers
        """

        # compile theano function, if haven't already done so
        if self.eval_us_f is None:
            self.eval_us_f = theano.function(
                inputs=[self.input],
                outputs=self.made.u
            )

        return self.eval_us_f(x.astype(dtype))


class ConditionalMaskedAutoregressiveFlow_on_MADE:
    """
    Conditional Masked Autoregressive Flow, where the target distribution is a conditional Mog MADE.
    """

    def __init__(self, n_inputs, n_outputs, n_hiddens, act_fun, n_layers, n_comps, batch_norm=True, output_order='sequential', mode='sequential', input=None, output=None):
        """
        Constructor.
        :param n_inputs: number of (conditional) inputs
        :param n_outputs: number of outputs
        :param n_hiddens: list with number of hidden units for each hidden layer
        :param act_fun: name of activation function
        :param n_layers: number of layers in the flow
        :param n_comps: number of gaussians for each conditional of target made
        :param batch_norm: whether to use batch normalization between mades in the flow
        :param output_order: order of outputs of last made
        :param mode: strategy for assigning degrees to hidden nodes: can be 'random' or 'sequential'
        :param input: theano variable to serve as input; if None, a new variable is created
        :param output: theano variable to serve as output; if None, a new variable is created
        """

        # save input arguments
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hiddens = n_hiddens
        self.act_fun = act_fun
        self.n_layers = n_layers
        self.n_comps = n_comps
        self.batch_norm = batch_norm
        self.mode = mode

        self.input = tt.matrix('x', dtype=dtype) if input is None else input
        self.y = tt.matrix('y', dtype=dtype) if output is None else output
        self.parms = []

        # maf
        self.maf = ConditionalMaskedAutoregressiveFlow(n_inputs, n_outputs, n_hiddens, act_fun, n_layers, batch_norm, output_order, mode, self.input, self.y)
        self.bns = self.maf.bns
        self.parms += self.maf.parms
        self.output_order = self.maf.output_order

        # mog made
        output_order = output_order if output_order == 'random' else self.maf.mades[-1].output_order[::-1]
        self.made = mades.ConditionalMixtureOfGaussiansMade(n_inputs, n_outputs, n_hiddens, act_fun, n_comps, output_order, mode, self.input, self.maf.u)
        self.parms += self.made.parms

        # log likelihoods
        self.L = self.made.L + self.maf.logdet_dudy
        self.L.name = 'L'

        # train objective
        self.trn_loss = -tt.mean(self.L)
        self.trn_loss.name = 'trn_loss'

        # theano evaluation functions, will be compiled when first needed
        self.eval_lprob_f = None
        self.eval_us_f = None

    def eval(self, xy, log=True):
        """
        Evaluate log probabilities for given input-output pairs.
        :param xy: a pair (x, y) where x rows are inputs and y rows are outputs
        :param log: whether to return probabilities in the log domain
        :return: log probabilities: log p(y|x)
        """

        # compile theano function, if haven't already done so
        if self.eval_lprob_f is None:
            self.eval_lprob_f = theano.function(
                inputs=[self.input, self.y],
                outputs=self.L,
                givens=[(bn.m, bn.bm) for bn in self.bns] + [(bn.v, bn.bv) for bn in self.bns]
            )

        x, y = xy
        lprob = self.eval_lprob_f(x.astype(dtype), y.astype(dtype))

        return lprob if log else np.exp(lprob)

    def gen(self, x, n_samples=1, u=None):
        """
        Generate samples, by propagating random numbers through each made, after conditioning on input x.
        :param x: input vector
        :param n_samples: number of samples
        :param u: random numbers to use in generating samples; if None, new random numbers are drawn
        :return: samples
        """

        y = rng.randn(n_samples, self.n_outputs).astype(dtype) if u is None else u
        y = self.made.gen(x, n_samples, y)
        y = self.maf.gen(x, n_samples, y)

        return y

    def calc_random_numbers(self, xy):
        """
        Givan a dataset, calculate the random numbers used internally to generate the dataset.
        :param xy: a pair (x, y) of numpy arrays, where x rows are inputs and y rows are outputs
        :return: numpy array, rows are corresponding random numbers
        """

        # compile theano function, if haven't already done so
        if self.eval_us_f is None:
            self.eval_us_f = theano.function(
                inputs=[self.input, self.y],
                outputs=self.made.u
            )

        x, y = xy
        return self.eval_us_f(x.astype(dtype), y.astype(dtype))
