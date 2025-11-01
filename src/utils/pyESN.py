import numpy as np


def correct_dimensions(s, targetlength):
    """checks the dimensionality of some numeric argument s, broadcasts it
       to the specified length if possible.

    Args:
        s: None, scalar or 1D array
        targetlength: expected length of s

    Returns:
        None if s is None, else numpy vector of length targetlength
    """
    if s is not None:
        s = np.array(s)
        if s.ndim == 0:
            s = np.array([s] * targetlength)
        elif s.ndim == 1:
            if not len(s) == targetlength:
                raise ValueError("arg must have length " + str(targetlength))
        else:
            raise ValueError("Invalid argument")
    return s


def identity(x):
    return x


class ESN():

    def __init__(self, n_inputs, n_outputs, n_reservoir=200,
                 spectral_radius=0.95, sparsity=0, noise=0.001, input_shift=None,
                 input_scaling=None, teacher_forcing=True, feedback_scaling=None,
                 teacher_scaling=None, teacher_shift=None,
                 out_activation=identity, inverse_out_activation=identity,
                 random_state=None, silent=True,
                 leaking_rate=1.0,           # α=1.0 reproduces your current behavior
                 ridge_lambda=1e-4):         # expose ridge (stronger default than 1e-6)
                
        """
        Args:
            n_inputs: nr of input dimensions
            n_outputs: nr of output dimensions
            n_reservoir: nr of reservoir neurons
            spectral_radius: spectral radius of the recurrent weight matrix
            sparsity: proportion of recurrent weights set to zero
            noise: noise added to each neuron (regularization)
            input_shift: scalar or vector of length n_inputs to add to each
                        input dimension before feeding it to the network.
            input_scaling: scalar or vector of length n_inputs to multiply
                        with each input dimension before feeding it to the netw.
            teacher_forcing: if True, feed the target back into output units
            teacher_scaling: factor applied to the target signal
            teacher_shift: additive term applied to the target signal
            out_activation: output activation function (applied to the readout)
            inverse_out_activation: inverse of the output activation function
            random_state: positive integer seed, np.rand.RandomState object,
                          or None to use numpy's builting RandomState.
            silent: supress messages
        """

        self.reservoir_states = list() #added this to track the reservoir states
        
        # check for proper dimensionality of all arguments and write them down.
        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.n_outputs = n_outputs
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.noise = noise
        self.input_shift = correct_dimensions(input_shift, n_inputs)
        self.input_scaling = correct_dimensions(input_scaling, n_inputs)

        self.teacher_scaling = teacher_scaling
        self.teacher_shift = teacher_shift

        self.out_activation = out_activation
        self.inverse_out_activation = inverse_out_activation
        self.random_state = random_state

        # the given random_state might be either an actual RandomState object,
        # a seed or None (in which case we use numpy's builtin RandomState)
        if isinstance(random_state, np.random.RandomState):
            self.random_state_ = random_state
        elif random_state:
            try:
                self.random_state_ = np.random.RandomState(random_state)
            except TypeError as e:
                raise Exception("Invalid seed: " + str(e))
        else:
            self.random_state_ = np.random.mtrand._rand

        self.teacher_forcing = teacher_forcing
        self.silent = silent
        self.leaking_rate   = float(leaking_rate)
        self.feedback_scaling = 1.0 if feedback_scaling is None else float(feedback_scaling)
        self.ridge_lambda   = float(ridge_lambda)
        self.initweights()

    def initweights(self):
        # initialize recurrent weights:
        # begin with a random matrix centered around zero:
        W = self.random_state_.rand(self.n_reservoir, self.n_reservoir) - 0.5
        # delete the fraction of connections given by (self.sparsity):
        W[self.random_state_.rand(*W.shape) < self.sparsity] = 0
        # compute the spectral radius of these weights:
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        # rescale them to reach the requested spectral radius:
        self.W = W * (self.spectral_radius / radius)

        # random input weights:
        self.W_in = self.random_state_.rand(
            self.n_reservoir, self.n_inputs) * 2 - 1
        # random feedback (teacher forcing) weights:
        self.W_feedb = self.random_state_.rand(
            self.n_reservoir, self.n_outputs) * 2 - 1

    def _update(self, state, input_pattern, output_pattern):
        if self.teacher_forcing:
            preactivation = (np.dot(self.W, state)
                            + np.dot(self.W_in, input_pattern)
                            + np.dot(self.W_feedb, self.feedback_scaling * output_pattern))
        else:
            preactivation = (np.dot(self.W, state)
                            + np.dot(self.W_in, input_pattern))

        candidate = np.tanh(preactivation)
        candidate += self.noise * (self.random_state_.rand(self.n_reservoir) - 0.5)
        new_state = (1.0 - self.leaking_rate) * state + self.leaking_rate * candidate

        self.reservoir_states.append(new_state)
        return new_state


    def _scale_inputs(self, inputs):
        """for each input dimension j: multiplies by the j'th entry in the
        input_scaling argument, then adds the j'th entry of the input_shift
        argument."""
        if self.input_scaling is not None:
            inputs = np.dot(inputs, np.diag(self.input_scaling))
        if self.input_shift is not None:
            inputs = inputs + self.input_shift
        return inputs

    def _scale_teacher(self, teacher):
        """multiplies the teacher/target signal by the teacher_scaling argument,
        then adds the teacher_shift argument to it."""
        if self.teacher_scaling is not None:
            teacher = teacher * self.teacher_scaling
        if self.teacher_shift is not None:
            teacher = teacher + self.teacher_shift
        return teacher

    def _unscale_teacher(self, teacher_scaled):
        """inverse operation of the _scale_teacher method."""
        if self.teacher_shift is not None:
            teacher_scaled = teacher_scaled - self.teacher_shift
        if self.teacher_scaling is not None:
            teacher_scaled = teacher_scaled / self.teacher_scaling
        return teacher_scaled


    def fit(self, inputs, outputs, inspect=False):
        """
        Collect the network's reaction to training data, train read-out weights.

        Args:
            inputs : array (T × n_inputs)  *or*  list/tuple of arrays
            outputs: array (T × n_outputs) *or*  list/tuple of arrays
            inspect: show a visualisation of the collected reservoir states

        Returns
            Model output on the training data.  For many sequences a list
            of arrays is returned (one per sequence); otherwise a single array.
        """
       
        single_sequence = not isinstance(inputs, (list, tuple))
        if single_sequence:
            inputs, outputs = [inputs], [outputs]

        if len(inputs) != len(outputs):
            raise ValueError("inputs and outputs must have the same length")

        extended_blocks, teacher_blocks, pred_blocks = [], [], []

        for in_raw, out_raw in zip(inputs, outputs):

            # transform any vectors of shape (x,) into (x,1):
            if in_raw.ndim < 2:
                in_raw = np.reshape(in_raw, (len(in_raw), -1))
            if out_raw.ndim < 2:
                out_raw = np.reshape(out_raw, (len(out_raw), -1))

            # scale
            inputs_scaled   = self._scale_inputs(in_raw)
            teachers_scaled = self._scale_teacher(out_raw)

            #if not self.silent:
            #    print("harvesting states...")
            states = np.zeros((inputs_scaled.shape[0], self.n_reservoir))
            for n in range(1, inputs_scaled.shape[0]):
                states[n, :] = self._update(states[n - 1],
                                            inputs_scaled[n, :],
                                            teachers_scaled[n - 1, :])

           
            n_eff = states.shape[0] - 1 
            
            n_eff = states.shape[0] - 1         
            extended_states = np.hstack([
                states[1:],               # reservoir states
                inputs_scaled[1:],        # current inputs
                np.ones((n_eff, 1))       # bias column
            ])


            extended_blocks.append(extended_states)
            teacher_blocks.append(teachers_scaled[1:])                  

            self.laststate  = states[-1, :].copy()
            self.lastinput  = in_raw[-1, :].copy()
            self.lastoutput = teachers_scaled[-1, :].copy()

        extended_states = np.vstack(extended_blocks)
        teachers_scaled = np.vstack(teacher_blocks)

        if not self.silent:
            print("fitting...")

            
        '''
        transient = min(int(extended_states.shape[0] / 10), 100)

        self.W_out = np.dot(np.linalg.pinv(extended_states[transient:, :]),
                            self.inverse_out_activation(
                                teachers_scaled[transient:, :])).T'''
        X_blocks   = []
        Y_blocks   = []
        for X_seq, Y_seq in zip(extended_blocks, teacher_blocks):
            k   = min(int(0.1*len(X_seq)), 100)     # 10 % or 100 samples
            X_blocks.append(X_seq[k:, :])
            Y_blocks.append(Y_seq[k:, :])
        X = np.vstack(X_blocks)
        Y = np.vstack(Y_blocks)
        

        reg = self.ridge_lambda
        XtX = X.T @ X + reg * np.eye(X.shape[1])
        self.W_out = (np.linalg.solve(XtX, X.T @ Y)).T

        if inspect:
            from matplotlib import pyplot as plt
            plt.figure(figsize=(extended_states.shape[0] * 0.0025,
                                extended_states.shape[1] * 0.01))
            plt.imshow(extended_states.T, aspect='auto', interpolation='nearest')
            plt.colorbar()

        if not self.silent:
            print("training error:")

        start = 0
        for X_seq, Y_seq in zip(extended_blocks, teacher_blocks):
            pred_seq = self._unscale_teacher(self.out_activation(
                         np.dot(X_seq, self.W_out.T)))
            pred_blocks.append(pred_seq)

            if not self.silent:
                err = np.sqrt(np.mean((pred_seq - self._unscale_teacher(Y_seq))**2))
                #print(err)

            start += X_seq.shape[0]

        self.reservoir_states = []  # reset tracker
        return pred_blocks[0] if single_sequence else pred_blocks


    
    def predict(self, inputs, continuation=True):
        """
        Apply the learned weights to the network's reactions to new input.

        Parameters
        ----------
        inputs : ndarray (T × n_inputs)           – original usage
                 list / tuple of ndarray          – several test trajectories
        continuation : bool, optional
            *Single-sequence case*  – if True, start from the last training state;
            *Multi-sequence case*   – the first sequence uses the last training
            state, all following ones start from zeros.

        Returns
        -------
        ndarray                  – single sequence (legacy behaviour)
        list of ndarray          – multiple sequences (one per input array)
        """

        single_sequence = not isinstance(inputs, (list, tuple))
        if single_sequence:
            inputs = [inputs]                      # wrap in list

        preds = []                                 # store outputs per sequence
        first_seq = True

        for in_raw in inputs:

            # reshape to 2-D if needed
            if in_raw.ndim < 2:
                in_raw = np.reshape(in_raw, (len(in_raw), -1))
            n_samples = in_raw.shape[0]

            # choose starting state/input/output
            if continuation and first_seq:
                laststate  = self.laststate
                lastinput  = self.lastinput
                lastoutput = self.lastoutput
            else:
                laststate  = np.zeros(self.n_reservoir)
                lastinput  = np.zeros(self.n_inputs)
                lastoutput = np.zeros(self.n_outputs)

            # prepend last input, scale new inputs
            in_scaled = self._scale_inputs(in_raw)
            inputs_aug = np.vstack([lastinput, in_scaled])

            states  = np.vstack([laststate,
                                 np.zeros((n_samples, self.n_reservoir))])
            outputs = np.vstack([lastoutput,
                                 np.zeros((n_samples, self.n_outputs))])

            for n in range(n_samples):
                states[n + 1, :] = self._update(states[n, :],
                                                inputs_aug[n + 1, :],
                                                outputs[n, :])
                
                vec = np.concatenate([states[n+1], inputs_aug[n+1], [1.0]])
                outputs[n+1] = self.out_activation(self.W_out @ vec)
                

            # remove prepended row, unscale, store prediction
            preds.append(self._unscale_teacher(outputs[1:]))
            #preds.append(self._unscale_teacher(self.out_activation(outputs[1:])))

            first_seq = False                      # subsequent seqs start from zero

        self.laststate  = states[-1, :].copy()
        self.lastinput  = inputs_aug[-1, :].copy()
        self.lastoutput = outputs[-1, :].copy()
        
        return preds[0] if single_sequence else preds

    def collect_reservoir_states(self, inputs):
        """
        Pass input points through the ESN reservoir and collect the reservoir states.
        No predictions are made, and only the reservoir states are returned.
    
        Args:
            inputs: array of shape (number_of_samples, 3) containing the input data points
    
        Returns:
            reservoir_states: array of collected reservoir states, shape (number_of_samples, n_reservoir)
        """
        if inputs.ndim < 2:
            inputs = np.reshape(inputs, (len(inputs), -1))
    
        n_samples = inputs.shape[0]  # Number of input points

        # Initialize the reservoir state (can be zeros or start from last known state)
        reservoir_state = np.zeros(self.n_reservoir)

        # Store the reservoir states
        reservoir_states = np.zeros((n_samples, self.n_reservoir))

        # Loop over the inputs, updating the reservoir state after each input
        for n in range(n_samples):
            reservoir_state = self._update(reservoir_state, inputs[n, :], np.zeros(self.n_outputs))
            reservoir_states[n, :] = reservoir_state  # Store the state for each input point

        return reservoir_states


