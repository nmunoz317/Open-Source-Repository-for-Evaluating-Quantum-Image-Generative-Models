import math
import typing
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import einops
import pennylane as qml
import tqdm
import qw_map

"""
MIT License

Copyright (c) 2024 Michael Kölle
"""

class PatchGAN():
    """
    Implementation of the quantum Patch GAN architecture [1]. The code was extracted from [2] and has been slightly modified.

    [1] Huang, H. L., Du, Y., Gong, M., Zhao, Y., Wu, Y., Wang, C., ... & Pan, J. W. (2021). Experimental quantum generative adversarial networks for image generation. Physical Review Applied, 16(2), 024051.

    [2] James Ellis. Quantum GANs. Aug. 2024. url: https://pennylane.ai/qml/demos/tutorial_quantum_gans/
    """
    def __init__(self, image_shape, n_generators, n_qubits, n_ancillas, n_layers, device, q_delta=1):
        self.discriminator = self.Discriminator(image_shape)
        self.device=device
        self.generator = self.Generator(n_generators, n_qubits, n_ancillas, n_layers, q_delta)

    class Discriminator(nn.Module):
        def __init__(self, image_shape):
            super().__init__()

            self.model = nn.Sequential(
                nn.Linear(int(np.prod(image_shape)), 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            return self.model(x)

    class Generator(nn.Module):
        def __init__(self, n_generators, n_qubits, n_ancillas, n_layers, q_delta):
            super().__init__()

            self.q_params = nn.ParameterList(
                [
                    nn.Parameter(q_delta * torch.rand(n_layers * n_qubits), requires_grad=True)
                    for _ in range(n_generators)
                ]
            )
            self.n_generators = n_generators
            self.n_qubits=n_qubits
            self.n_a_qubits=n_ancillas
            self.q_depth=n_layers
            self.q_dev = qml.device("default.qubit", wires=n_qubits)
            self.qnode=qml.QNode(self.quantum_circuit, self.q_dev, interface="torch", diff_method="parameter-shift")

        def forward(self, x):
            patch_size = 2 ** (self.n_qubits - self.n_a_qubits)
            images = torch.Tensor(x.size(0), 0).to(self.device)
            for params in self.q_params:
                patches = torch.Tensor(0, patch_size).to(self.device)
                for elem in x:
                    q_out = self.partial_measure(elem, params).float().unsqueeze(0)
                    patches = torch.cat((patches, q_out))

                images = torch.cat((images, patches), 1)

            return images
        
        def quantum_circuit(self, noise, weights):

            weights = weights.reshape(self.q_depth, self.n_qubits)

            for i in range(self.n_qubits):
                qml.RY(noise[i], wires=i)

            for i in range(self.q_depth):
                for y in range(self.n_qubits):
                    qml.RY(weights[i][y], wires=y)
                for y in range(self.n_qubits - 1):
                    qml.CZ(wires=[y, y + 1])

            return qml.probs(wires=list(range(self.n_qubits)))

        def partial_measure(self, noise, weights):
            probs = self.qnode(noise, weights)
            probsgiven0 = probs[: (2 ** (self.n_qubits - self.n_a_qubits))]
            probsgiven0 /= torch.sum(probs)
            probsgiven = probsgiven0 / torch.max(probsgiven0)
            return probsgiven

class QCGAN():
    """
    Implementation of the QC-GAN architecture [1]. 

    [1] Shu, R., Xu, X., Yung, M. H., & Cui, W. (2024). Variational Quantum Circuits Enhanced Generative Adversarial Network. arXiv preprint arXiv:2402.01791.
    """
    def __init__(self, image_shape, n_qubits, n_layers, q_delta=1, device='cpu'):
        self.discriminator = self.Discriminator(image_shape)
        self.device=device
        self.generator = self.Generator(n_layers, n_qubits, image_shape, q_delta)
    class Discriminator(nn.Module):
            def __init__(self, image_shape):
                super().__init__()
                self.model = nn.Sequential(
                    nn.Linear(np.prod(image_shape), 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1),
                    nn.Sigmoid(),
                )

            def forward(self, x):
                return self.model(x)
    class Generator(nn.Module):

        def __init__(self, layers, n_data_qubits, image_shape, q_delta=1):

            super().__init__()

            self.q_params = nn.Parameter(q_delta * torch.rand(layers * n_data_qubits*3), requires_grad=True)
            self.q_dev = qml.device("default.qubit", wires=n_data_qubits)
            self.qnode=qml.QNode(self.quantum_circuit, self.q_dev, interface="torch", diff_method="parameter-shift")
            self.linear = nn.Linear(2**n_data_qubits,np.prod(image_shape))
            self.sigmoid = nn.Sigmoid()
            self.layers=layers
            self.n_data_qubits=n_data_qubits
            self.image_shape=image_shape

        def forward(self, x):
            outputs = []
            for elem in x:  
                q_out = self.partial_measure(elem, self.q_params).float().unsqueeze(0)
                outputs.append(q_out)  
            images = torch.cat(outputs, dim=0).to(self.device)

            linear_output = self.linear(images)

            output = self.sigmoid(linear_output)

            return output.view(-1, 1, *self.image_shape)

        def partial_measure(self,noise, weights):
            probs = self.qnode(noise, weights).float()
            quantum_output = probs / torch.abs(probs).max()

            return quantum_output
        def quantum_circuit(self,noise, weights):

          weights = weights.reshape(self.layers, self.n_data_qubits,3)

          for i in range(self.n_data_qubits):
              qml.RY(noise[i], wires=i)

          for i in range(self.layers):

              for y in range(self.n_data_qubits):
                  qml.RX(weights[i][y][0], wires=y)
                  qml.RZ(weights[i][y][1], wires=y)

              
              for y in range(self.n_data_qubits - 1):
                  qml.CRX(weights[i, y, 2], wires=[y, y + 1])
              qml.CRX(weights[i, self.n_data_qubits - 1, 2], wires=[(self.n_data_qubits-1), 0])

          return qml.probs(wires=list(range(self.n_data_qubits)))


class PQWGAN():
    """
    Implementation of the PQWGAN architecture [1]. The code was extracted from [2] and has been slightly modified.

    [1] Tsang, S. L., West, M. T., Erfani, S. M., & Usman, M. (2023). Hybrid quantum-classical generative adversarial network for high resolution image generation. IEEE Transactions on Quantum Engineering.

    [2] Jasontslxd. GitHub - jasontslxd/PQWGAN: Code to run and test the PQWGAN framework. url: https://github.com/jasontslxd/PQWGAN
    """
    def __init__(self, image_size, channels, n_generators, n_qubits, n_ancillas, n_layers, patch_shape, device):
        self.image_shape = (channels, image_size[0], image_size[1])
        self.device=device
        self.discriminator = self.Discriminator(self.image_shape)
        self.generator = self.Generator(n_generators, n_qubits, n_ancillas, n_layers, self.image_shape, patch_shape)

    class Discriminator(nn.Module):
        def __init__(self, image_shape):
            super().__init__()
            self.image_shape = image_shape

            self.fc1 = nn.Linear(int(np.prod(self.image_shape)), 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 1)

        def forward(self, x):
            x = x.view(x.shape[0], -1)
            x = F.leaky_relu(self.fc1(x), 0.2).to(self.device)
            x = F.leaky_relu(self.fc2(x), 0.2).to(self.device)
            return self.fc3(x)

    class Generator(nn.Module):
        def __init__(self, n_generators, n_qubits, n_ancillas, n_layers, image_shape, patch_shape):
            super().__init__()
            self.n_generators = n_generators
            self.n_qubits = n_qubits
            self.n_ancillas = n_ancillas
            self.n_layers = n_layers
            self.q_device = qml.device("default.qubit", wires=n_qubits)
            self.params = nn.ParameterList([nn.Parameter(torch.rand(n_layers, n_qubits, 3), requires_grad=True) for _ in range(n_generators)])
            self.qnode = qml.QNode(self.circuit, self.q_device, interface="torch")

            self.image_shape = image_shape
            self.patch_shape = patch_shape

        def forward(self, x):
            special_shape = bool(self.patch_shape[0]) and bool(self.patch_shape[1])
            patch_size = 2 ** (self.n_qubits - self.n_ancillas)
            image_pixels = self.image_shape[2] ** 2
            pixels_per_patch = image_pixels // self.n_generators
            if special_shape and self.patch_shape[0] * self.patch_shape[1] != pixels_per_patch:
                raise ValueError("patch shape and patch size dont match!")
            output_images = torch.Tensor(x.size(0), 0).to(self.device)
            for sub_generator_param in self.params:
                patches = torch.Tensor(0, pixels_per_patch).to(self.device)
                for item in x:
                    sub_generator_out = self.partial_trace_and_postprocess(item, sub_generator_param).float().unsqueeze(0)
                    if pixels_per_patch < patch_size:
                        sub_generator_out = sub_generator_out[:,:pixels_per_patch].to(self.device)
                    patches = torch.cat((patches, sub_generator_out)).to(self.device)
                output_images = torch.cat((output_images, patches), 1).to(self.device)

            if special_shape:
                final_out = torch.zeros(x.size(0), *self.image_shape)
                for i,img in enumerate(output_images):
                    for patches_done, j in enumerate(range(0, img.shape[0], pixels_per_patch)):
                        patch = torch.reshape(img[j:j+pixels_per_patch], self.patch_shape)
                        starting_h = ((patches_done * self.patch_shape[1]) // self.image_shape[2]) * self.patch_shape[0]
                        starting_w = (patches_done * self.patch_shape[1]) % self.image_shape[2]
                        final_out[i, 0, starting_h:starting_h+self.patch_shape[0], starting_w:starting_w+self.patch_shape[1]] = patch
            else:
                final_out = output_images.view(output_images.shape[0], *self.image_shape)
            return final_out

        def circuit(self, latent_vector, weights):
            for i in range(self.n_qubits):
                qml.RY(latent_vector[i], wires=i)
            
            for i in range(self.n_layers):
                for j in range(self.n_qubits):
                    qml.Rot(*weights[i][j], wires=j)

                for j in range(self.n_qubits-1):
                    qml.CNOT(wires=[j, j+1])
            
            return qml.probs(wires=list(range(self.n_qubits)))

        def partial_trace_and_postprocess(self, latent_vector, weights):
            probs = self.qnode(latent_vector, weights)
            probs_given_ancilla_0 = probs[:2**(self.n_qubits - self.n_ancillas)]
            post_measurement_probs = probs_given_ancilla_0 / torch.sum(probs_given_ancilla_0)
            post_processed_patch = ((post_measurement_probs / torch.max(post_measurement_probs)) - 0.5) * 2
            return post_processed_patch

class GAN():
    """
    Implementation of a classic GAN architecture. The code was extracted from [1] and has been slightly modified.

    [1] Real Python. Generative adversarial networks: Build your first models. Aug. 2024. url: https://realpython.com/generative-adversarial-networks/.
    """
    def __init__(self, img_shape, device):
        self.generator = self.Generator(img_shape)
        self.device=device
        self.discriminator = self.Discriminator(img_shape)

    class Generator(nn.Module):
        def __init__(self, img_shape):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(100, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Linear(1024, int(np.prod(img_shape))),
                nn.Tanh(),
            )
            self.img_shape=img_shape

        def forward(self, x):
            output = self.model(x)
            output = output.view(x.size(0), 1, self.img_shape[0], self.img_shape[1])
            return output

    class Discriminator(nn.Module):
        def __init__(self, img_shape):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(int(np.prod(img_shape)), 1024),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 1),
                nn.Sigmoid(),
            )
            self.img_shape=img_shape

        def forward(self, x):
            x = x.view(x.size(0), int(np.prod(self.img_shape)))
            output = self.model(x)
            return output

class WGAN():
    """
    Implementation of the WGAN architecture [1]. The code was extracted from [2] and has been slightly modified.

    [1] Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., & Courville, A. C. (2017). Improved training of wasserstein gans. Advances in neural information processing systems, 30.

    [2] Eriklindernoren. PyTorch-GAN. url: https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/wgan_gp.
    """
    def __init__(self, img_shape, device):
        self.generator = self.Generator(img_shape)
        self.device=device
        self.discriminator = self.Discriminator(img_shape)

    class Generator(nn.Module):
        def __init__(self, img_shape):
            super().__init__()
            self.img_shape = img_shape

            def block(in_feat, out_feat, normalize=True):
                layers = [nn.Linear(in_feat, out_feat)]
                if normalize:
                    layers.append(nn.BatchNorm1d(out_feat, 0.8))
                layers.append(nn.LeakyReLU(0.2, inplace=True))
                return layers

            self.model = nn.Sequential(
                *block(100, 128, normalize=False),
                *block(128, 256),
                *block(256, 512),
                *block(512, 1024),
                nn.Linear(1024, int(np.prod(img_shape))),
                nn.Tanh()
            )

        def forward(self, z):
            img = self.model(z)
            img = img.view(img.shape[0], *self.img_shape)
            img=img.unsqueeze(1)
            return img

    class Discriminator(nn.Module):
        def __init__(self, img_shape):
            super().__init__()
            self.img_shape = img_shape

            self.model = nn.Sequential(
                nn.Linear(int(np.prod(img_shape)), 512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 1),
            )

        def forward(self, img):
            img_flat = img.view(img.shape[0], -1)
            validity = self.model(img_flat)
            return validity



"""
    Implementation of the Q-Dense and Q-Dense Directed architectures [1]. The code was extracted from [2] and has been slightly modified.
    
    Based on software copyrighted by Michael Kölle, licensed under the MIT License.
    
    [1] Michael Kölle et al. “Quantum Denoising Diffusion Models”. In: arXiv preprint arXiv:2401.07049 (2024).

    [2] Michaelkoelle. GitHub - michaelkoelle/quantum-diffusion: Quantum Denoising Diffusion Models. url: https://github.com/michaelkoelle/quantum-diffusion.
"""
class Diffusion(torch.nn.Module):
    def __init__(
        self,
        net: torch.nn.Module,
        noise_f,
        prediction_goal: str,
        shape: typing.Tuple[int, int],
        loss: torch.nn.Module = torch.nn.MSELoss(reduction="none"),
        directed=False,
        on_states=False,
        skip_name_test=False,
    ) -> None:
        super().__init__()
        self.net = net
        if not skip_name_test:
            assert (
                net._get_name().lower().__contains__("directed")
            ), f"Unknown network type {net._get_name()}"
            if directed and net._get_name().lower().__contains__("undirected"):
                raise ValueError(
                    f"Directed model cannot be used with undirected network {net._get_name()}"
                )
            if not directed and not net._get_name().lower().__contains__("undirected"):
                raise ValueError(
                    f"Undirected model cannot be used with directed network {net._get_name()}"
                )
        assert prediction_goal in [
            "data",
            "noise",
        ], "prediction_goal must be either 'data' or 'noise'"
        self.prediction_goal = prediction_goal
        self.add_noise = noise_f
        self.width, self.height = shape
        self.loss = loss
        self.directed = directed
        self.on_states = on_states
        if self.on_states:
            self.loss = StateLoss()

    def forward(
        self, x: typing.Union[torch.Tensor, None], **kwargs
    ) -> typing.Union[torch.Tensor, None]:
        x = typing.cast(torch.Tensor, x)
        return self.run_training_step_data(x, **kwargs)
            
    def run_training_step_data(self, x: torch.Tensor, **kwargs) -> typing.Any:
        assert self.prediction_goal == "data", "prediction goal is not data"
        T = kwargs["T"]
        if self.directed:
            y = kwargs["y"]
            assert y.shape[0] == x.shape[0], "batch size of x and y must be the same"
            labels = einops.repeat(y, "b -> (b tau)", tau=T)
        else:
            if "y" in kwargs and kwargs["y"] is not None:
                warnings.warn("y is not used because the model is not directed")
        whole_noisy = self.add_noise(x, tau=T + 1, decay_mod=3.0)
        whole_noisy = einops.rearrange(
            whole_noisy, "(batch tau) pixels -> batch tau pixels", tau=T + 1
        )
        batches_noisy = whole_noisy[:, 1:, :]
        batches_clean = whole_noisy[:, :-1, :]
        batches_noisy = einops.rearrange(
            batches_noisy,
            "batch tau (width height) -> (batch tau) 1 width height",
            width=self.width,
            height=self.height,
        )
        batches_clean = einops.rearrange(
            batches_clean,
            "batch tau (width height) -> (batch tau) 1 width height",
            width=self.width,
            height=self.height,
        )
        if self.directed:
            batches_reconstructed = self.net.forward(x=batches_noisy, y=labels) 
        else:
            batches_reconstructed = self.net.forward(x=batches_noisy)
        batch_loss = self.loss(batches_reconstructed, batches_clean)
        batch_loss_mean = batch_loss.mean()
        batch_loss_mean.backward()
        verbose = kwargs.get("verbose", False)
        if verbose:
            return batch_loss.abs(), batches_reconstructed.abs()
        else:
            return (batch_loss_mean.abs(),)


    def sample(
        self,
        n_iters,
        first_x: typing.Union[torch.Tensor, None] = None,
        labels: typing.Union[torch.Tensor, None] = None,
        show_progress: bool = False,
        only_last=False,
        step=1,
        noise_factor=1.0,
    ) -> torch.Tensor:
        """ " Samples from the model for n_iters iterations."""
        if first_x is None:
            first_x = torch.rand((10, 1, self.width, self.height))
        if self.on_states:
            return self._sample_on_states(n_iters, first_x, only_last, labels=labels)
        if labels is None and self.directed:
            labels = torch.zeros((first_x.shape[0], 1))
        if labels is not None and not self.directed:
            warnings.warn("labels are not used because the model is not directed")
        outp = [first_x]
        if show_progress:
            iters = tqdm.tqdm(range(n_iters))
        else:
            iters = range(n_iters)
        with torch.no_grad():
            x = first_x
            for i in iters:
                if self.directed:
                    predicted = self.net(x, labels)
                else:
                    predicted = self.net(x)
                if self.prediction_goal == "data":
                    x = predicted
                else:
                    predicted = (predicted - 0.5) * 0.1 * noise_factor
                    new_x = x - predicted
                    new_x = torch.clamp(new_x, 0, 1)
                    x = new_x
                if i % step == 0:
                    outp.append(x)
        if only_last:
            return outp[-1]
        else:
            outp = torch.stack(outp)
            outp = einops.rearrange(
                outp, "iters batch 1 height width -> (iters height) (batch width)"
            )
            return outp

    def _sample_on_states(
        self, n_iters: int, first_x: torch.Tensor, only_last=True, labels=None
    ) -> torch.Tensor:
        assert only_last, "can't sample intermediate states, set `only_last=True`"
        assert self.prediction_goal == "data", "can't sample noise"
        assert self.on_states, "use sample() instead"
        return self.net.sample(first_x, num_repeats=n_iters, labels=labels)  # type: ignore

    def get_variance_sample(self, **kwargs) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        sample = self.sample(**kwargs).abs()
        sample = einops.rearrange(
            sample,
            "(iters height) (batch width) -> iters batch height width",
            height=self.height,
            width=self.width,
        )
        vars = sample.var(dim=1)
        sample = einops.rearrange(
            sample, "iters batch height width -> (iters height) (batch width)"
        )
        vars = einops.rearrange(vars, "iters height width -> (iters height) (width)")
        return sample, vars

    def save_name(self):
        return f"{self.net.save_name()}{'_noise' if self.prediction_goal == 'noise' else ''}"  # type: ignore


class StateLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert input.is_complex(), "input must be complex"
        assert not target.is_complex(), "target must be real"
        return (input.real - target) ** 2 + input.imag**2

class QDenseUndirected(torch.nn.Module):

    def __init__(self, qdepth, shape) -> None:
        super().__init__()
        self.qdepth = qdepth
        if isinstance(shape, int):
            shape = (32, 32)
        self.width, self.height = 32,32
        self.pixels = self.width * self.height
        self.wires = math.ceil(math.log2(self.width * self.height))
        self.qdev = qml.device("default.qubit", wires=self.wires)
        weight_shape = qml.StronglyEntanglingLayers.shape(self.qdepth, self.wires)
        self.weights = torch.nn.Parameter(
            torch.randn(weight_shape, requires_grad=True) * 0.4
        )
        self.qnode = qml.QNode(
            func=self._circuit,
            device=self.qdev,
            interface="torch",
            diff_method="backprop",
        )

    def _circuit(self, inp):
        qml.AmplitudeEmbedding(
            features=inp, wires=range(self.wires), normalize=True, pad_with=0.1
        )
        qml.StronglyEntanglingLayers(
            weights=qw_map.tanh(self.weights), wires=range(self.wires)
        )
        return qml.probs(wires=range(self.wires))

    def _post_process(self, probs):
        probs = probs[:, : self.pixels]
        probs = probs * self.pixels
        probs = torch.clamp(probs, 0, 1)
        return probs

    def forward(self, x):
        x = einops.rearrange(x, "b 1 w h -> b (w h)")
        x = self.qnode(x)
        x = self._post_process(x)
        x = einops.rearrange(x, "b (w h) -> b 1 w h", w=self.width, h=self.height)
        return x

    def __repr__(self):
        return f"QDenseUndirected(qdepth={self.qdepth}, wires={self.wires})"

    def save_name(self) -> str:
        return f"qdense_undirected_d{self.qdepth}_w{self.width}_h{self.height}"



class QDense2Undirected(torch.nn.Module):
    def __init__(
        self, qdepth, shape, entangling_layer=qml.StronglyEntanglingLayers
    ) -> None:
        super().__init__()
        self.qdepth = qdepth
        if isinstance(shape, int):
            shape = (shape, shape)
        self.width, self.height = shape
        self.pixels = self.width * self.height
        self.entangling_layer = entangling_layer
        self.wires = math.ceil(math.log2(self.width * self.height)) + 1
        self.qdev = qml.device("default.qubit", wires=self.wires)
        weight_shape = self.entangling_layer.shape(self.qdepth, self.wires)
        self.weights = torch.nn.Parameter(
            torch.randn(weight_shape, requires_grad=True) * 0.4
        )
        self.qnode = qml.QNode(
            func=self._circuit,
            device=self.qdev,
            interface="torch",
            diff_method="backprop",
        )

    def _circuit(self, inp):
        qml.AmplitudeEmbedding(
            features=inp, wires=range(self.wires - 1), normalize=True, pad_with=0.1
        )
        self.entangling_layer(
            weights=qw_map.tanh(self.weights), wires=range(self.wires)
        )
        return qml.probs(wires=range(self.wires))

    def _post_process(self, probs):
        probs = probs[:, ::2] 
        probs = probs[:, : self.pixels]
        probs = probs * self.pixels
        probs = torch.clamp(probs, 0, 1)
        return probs

    def forward(self, x):
        x = einops.rearrange(x, "b 1 w h -> b (w h)")
        x = self.qnode(x)
        x = self._post_process(x)
        x = einops.rearrange(x, "b (w h) -> b 1 w h", w=self.width, h=self.height)
        return x

    def __strongly(self):
        return "" if "strongly" in str(self.entangling_layer).lower() else "_weakly"

    def __repr__(self):
        return f"QDense2Undirected(qdepth={self.qdepth}, wires={self.wires}{self.__strongly()})"

    def save_name(self) -> str:
        return f"qdense2_undirected_d{self.qdepth}_w{self.width}_h{self.height}{self.__strongly()}"


class QDenseDirected(QDense2Undirected):
    def _circuit(self, inp, label):
        qml.AmplitudeEmbedding(
            features=inp, wires=range(self.wires - 1), normalize=True, pad_with=0.1
        )
        qml.RX(phi=label, wires=self.wires - 1)
        qml.StronglyEntanglingLayers(
            weights=qw_map.tanh(self.weights), wires=range(self.wires)
        )
        return qml.probs(wires=range(self.wires))

    def forward(self, x, y):
        x = einops.rearrange(x, "b 1 w h -> b (w h)")
        y=y.squeeze()
        x = self.qnode(x, y)
        x = self._post_process(x)
        x = einops.rearrange(x, "b (w h) -> b 1 w h", w=self.width, h=self.height)
        return x

    def __repr__(self):
        return f"QDenseDirected(qdepth={self.qdepth}, wires={self.wires})"

    def save_name(self) -> str:
        return f"qdense_directed_d{self.qdepth}_w{self.width}_h{self.height}"


"""
    Implementation of the Denoising Diffusion Probabilistic architecture. The code was extracted from [1] and has been slightly modified.

    [1] Brian Pulfer. “Generating images with DDPMs: A PyTorch Implementation”. In: (Mar. 2023). url: https://medium.com/@brianpulfer/enerating-images-with-ddpms-a-pytorch-implementation-cef5a2ba8cb1.
"""
def sinusoidal_embedding(n, d):
    embedding = torch.zeros(n, d)
    wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)])
    wk = wk.reshape((1, d))
    t = torch.arange(n).reshape((n, 1))
    embedding[:,::2] = torch.sin(t * wk[:,::2])
    embedding[:,1::2] = torch.cos(t * wk[:,::2])

    return embedding

class MyBlock(nn.Module):
    def __init__(self, shape, in_c, out_c, kernel_size=3, stride=1, padding=1, activation=None, normalize=True):
        super(MyBlock, self).__init__()
        self.ln = nn.LayerNorm(shape)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size, stride, padding)
        self.activation = nn.SiLU() if activation is None else activation
        self.normalize = normalize

    def forward(self, x):
        out = self.ln(x) if self.normalize else x
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.activation(out)
        return out
    
class MyUNet(nn.Module):
    def __init__(self, n_steps=1000, time_emb_dim=100):
        super(MyUNet, self).__init__()
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)
        self.te1 = self._make_te(time_emb_dim, 1)
        self.b1 = nn.Sequential(
            MyBlock((1, 32, 32), 1, 10),
            MyBlock((10, 32, 32), 10, 10),
            MyBlock((10, 32, 32), 10, 10)
        )
        self.down1 = nn.Conv2d(10, 10, 4, 2, 1)

        self.te2 = self._make_te(time_emb_dim, 10)
        self.b2 = nn.Sequential(
            MyBlock((10, 16, 16), 10, 20),
            MyBlock((20, 16, 16), 20, 20),
            MyBlock((20, 16, 16), 20, 20)
        )
        self.down2 = nn.Conv2d(20, 20, 4, 2, 1)

        self.te3 = self._make_te(time_emb_dim, 20)
        self.b3 = nn.Sequential(
            MyBlock((20, 8, 8), 20, 40),
            MyBlock((40, 8, 8), 40, 40),
            MyBlock((40, 8, 8), 40, 40)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(40, 40, 2, 1,1),
            nn.SiLU(),
            nn.Conv2d(40, 40, 4, 2, 1)
        )

        self.te_mid = self._make_te(time_emb_dim, 40)
        self.b_mid = nn.Sequential(
            MyBlock((40, 4, 4), 40, 20),
            MyBlock((20, 4, 4), 20, 20),
            MyBlock((20, 4, 4), 20, 40)
        )

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(40, 40, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(40, 40, 3, 1, 1)
        )

        self.te4 = self._make_te(time_emb_dim, 80)
        self.b4 = nn.Sequential(
            MyBlock((80, 8, 8), 80, 40),
            MyBlock((40, 8, 8), 40, 20),
            MyBlock((20, 8, 8), 20, 20)
        )

        self.up2 = nn.ConvTranspose2d(20, 20, 4, 2, 1)
        self.te5 = self._make_te(time_emb_dim, 40)
        self.b5 = nn.Sequential(
            MyBlock((40, 16, 16), 40, 20),
            MyBlock((20, 16, 16), 20, 10),
            MyBlock((10, 16, 16), 10, 10)
        )

        self.up3 = nn.ConvTranspose2d(10, 10, 4, 2, 1)
        self.te_out = self._make_te(time_emb_dim, 20)
        self.b_out = nn.Sequential(
            MyBlock((20, 32, 32), 20, 10),
            MyBlock((10, 32, 32), 10, 10),
            MyBlock((10, 32, 32), 10, 10, normalize=False)
        )

        self.conv_out = nn.Conv2d(10, 1, 3, 1, 1)

    def forward(self, x, t):
        t = self.time_embed(t)
        n = len(x)
        out1 = self.b1(x + self.te1(t).reshape(n, -1, 1, 1)) 
        out2 = self.b2(self.down1(out1) + self.te2(t).reshape(n, -1, 1, 1))  
        out3 = self.b3(self.down2(out2) + self.te3(t).reshape(n, -1, 1, 1))  

        out_mid = self.b_mid(self.down3(out3) + self.te_mid(t).reshape(n, -1, 1, 1))  

        out4 = torch.cat((out3, self.up1(out_mid)), dim=1)  
        out4 = self.b4(out4 + self.te4(t).reshape(n, -1, 1, 1))  

        out5 = torch.cat((out2, self.up2(out4)), dim=1)  
        out5 = self.b5(out5 + self.te5(t).reshape(n, -1, 1, 1))  

        out = torch.cat((out1, self.up3(out5)), dim=1)  
        out = self.b_out(out + self.te_out(t).reshape(n, -1, 1, 1))  

        out = self.conv_out(out)

        return out

    def _make_te(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out)
        )


class MyDDPM(nn.Module):
    def __init__(self, network, n_steps=200, min_beta=10 ** -4, max_beta=0.02, device=None):
        super(MyDDPM, self).__init__()
        self.n_steps = n_steps
        self.device = device
        self.network = network.to(device)
        self.betas = torch.linspace(min_beta, max_beta, n_steps).to(
            device)  
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))]).to(device)

    def forward(self, x0, t, eta=None):
        n, c, h, w = x0.shape
        a_bar = self.alpha_bars[t]

        if eta is None:
            eta = torch.randn(n, c, h, w).to(self.device)

        noisy = a_bar.sqrt().reshape(n, 1, 1, 1) * x0 + (1 - a_bar).sqrt().reshape(n, 1, 1, 1) * eta
        return noisy

    def backward(self, x, t):
        return self.network(x, t)
    def sample(self, first_x, n_iters, show_progress=False):
        outp=[first_x]
        if show_progress:
            iters = tqdm.tqdm(range(n_iters))
        else:
            iters = range(n_iters)
        with torch.no_grad():
            x=first_x
            for i in iters:
                t = torch.full((x.shape[0], 1), i)
                eta_theta = self.backward(x, t)
                alpha_t = self.alphas[i]
                alpha_t_bar = self.alpha_bars[i]
                x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)
                outp.append(x)
            outp = torch.stack(outp)
            outp = einops.rearrange(
                outp, "iters batch 1 height width -> (iters height) (batch width)"
            )
            return outp
                
        return x

