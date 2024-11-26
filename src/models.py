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



class PatchGAN():
    """
    Implementation of the quantum Patch GAN architecture [1]. The source code was extracted from [2].

    [1] Huang, H. L., Du, Y., Gong, M., Zhao, Y., Wu, Y., Wang, C., ... & Pan, J. W. (2021). Experimental quantum generative adversarial networks for image generation. Physical Review Applied, 16(2), 024051.

    [2] James Ellis. Quantum GANs. Aug. 2024. url: https://pennylane.ai/qml/demos/tutorial_quantum_gans/
    """
    def __init__(self, image_shape, n_generators, n_qubits, n_ancillas, n_layers, device, q_delta=1):
        self.discriminator = self.Discriminator(image_shape)
        self.device=device
        self.generator = self.Generator(n_generators, n_qubits, n_ancillas, n_layers, q_delta, device)

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
        def __init__(self, n_generators, n_qubits, n_ancillas, n_layers, q_delta, device):
            super().__init__()
            self.device=device
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
        self.generator = self.Generator(n_layers, n_qubits, image_shape, q_delta, device)
    class Discriminator(nn.Module):
            def __init__(self, image_shape):
                super().__init__()
                self.model = nn.Sequential(
                    nn.Linear(image_shape[0]*image_shape[1], 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1),
                    nn.Sigmoid(),
                )

            def forward(self, x):
                return self.model(x)
    class Generator(nn.Module):

        def __init__(self, layers, n_data_qubits, image_shape, q_delta=1, device='cuda'):

            super().__init__()
            self.device=device
            self.q_params = nn.Parameter(q_delta * torch.rand(layers * n_data_qubits*3), requires_grad=True)
            self.q_dev = qml.device("default.qubit", wires=n_data_qubits)
            self.qnode=qml.QNode(self.quantum_circuit, self.q_dev, interface="torch", diff_method="parameter-shift")
            self.linear = nn.Linear(2**n_data_qubits,np.prod(image_shape))
            self.sigmoid = nn.Sigmoid()
            self.layers=layers
            self.n_data_qubits=n_data_qubits
            self.image_shape=image_shape
        def forward(self, x):
            images = []

            for elem in x:
                q_out = self.partial_measure(elem, self.q_params).float().unsqueeze(0)
                images.append(q_out)

            images = torch.cat(images, dim=0).to(self.device)

      
            linear_output = self.linear(images)

            
            output = self.sigmoid(linear_output)

            return output

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
    Implementation of the PQWGAN architecture [1]. The source code was extracted from [2].

    [1] Tsang, S. L., West, M. T., Erfani, S. M., & Usman, M. (2023). Hybrid quantum-classical generative adversarial network for high resolution image generation. IEEE Transactions on Quantum Engineering.

    [2] Jasontslxd. GitHub - jasontslxd/PQWGAN: Code to run and test the PQWGAN framework. url: https://github.com/jasontslxd/PQWGAN
    """
    def __init__(self, image_size, channels, n_generators, n_qubits, n_ancillas, n_layers, patch_shape, device):
        self.image_shape = (channels, image_size[0], image_size[1])
        self.device=device
        self.discriminator = self.Discriminator(self.image_shape, device)
        self.generator = self.Generator(n_generators, n_qubits, n_ancillas, n_layers, self.image_shape, patch_shape, device)

    class Discriminator(nn.Module):
        def __init__(self, image_shape, device):
            super().__init__()
            self.image_shape = image_shape
            self.device=device
            self.fc1 = nn.Linear(int(np.prod(self.image_shape)), 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 1)

        def forward(self, x):
            x = x.view(x.shape[0], -1)
            x = F.leaky_relu(self.fc1(x), 0.2).to(self.device)
            x = F.leaky_relu(self.fc2(x), 0.2).to(self.device)
            return self.fc3(x)

    class Generator(nn.Module):
        def __init__(self, n_generators, n_qubits, n_ancillas, n_layers, image_shape, patch_shape, device):
            super().__init__()
            self.n_generators = n_generators
            self.device=device
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
    Implementation of a classic GAN architecture. The sourced code was extracted from [1].

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
    Implementation of the WGAN architecture [1]. The source code was extracted from [2].

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
    
    MIT License
    
    Copyright (c) 2024 Michael Kölle

    Implementation of the Q-Dense and Q-Dense Directed architectures [1]. The source code was extracted from [2].
    
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
            shape = (shape, shape)
        self.width, self.height = shape[0],shape[1]
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




