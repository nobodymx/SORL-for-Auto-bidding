import gin
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sdk.Common.Utils import normalize_state, softmax
from copy import deepcopy


@gin.configurable
class OnlineExplorationPolicy:
    def __init__(self, phi=None, u_v=None, sigma=1, lamda=10, sample_num=100, sample_range=3,
                 sample_way="weighted"):
        """
        :param phi: potential functions/classes, Q or A are natural choices, -- a function of s and a, using phi.cal()
        :param u_v: interaction policy -- a function of s, using u_v.cal()
        """
        self.phi = phi
        self.u_v = u_v
        self.sigma = sigma
        self.lamda = lamda

        # history parameters
        self.hist_phi = []
        self.hist_lamda = []

        # sampling way
        self.sampling_way = sample_way
        self.sampling_num = sample_num
        self.sampling_range = sample_range
        
        self.use_cuda = torch.cuda.is_available()
        self.FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
    
    def load_nets(self, phi_path=None, u_v_path=None):
        self.phi = torch.load(phi_path, map_location="cpu")
        self.u_v = torch.load(u_v_path, map_location="cpu")
        # cuda usage
        if self.use_cuda:
            self.phi.cuda()
            self.u_v.cuda()

    def explore_action(self, state, interact_action=None, draw=False, explore_flag="on"):
        """
        generating exploration actions
        :param interact_action: alternative, if None: use self.u_v to generate base action
        :param state: current state
        :return:
        """

        mean_action = self.u_v(
            normalize_state(torch.Tensor(deepcopy(state)).type(self.FloatTensor))) if interact_action is None else interact_action

        if explore_flag == "on":
            if self.sampling_way == "weighted":
                sample_actions = np.linspace(mean_action.detach().cpu().numpy() - self.sampling_range,
                                            mean_action.detach().cpu().numpy() + self.sampling_range,
                                            self.sampling_num)
                sample_actions = torch.Tensor(sample_actions).type(self.FloatTensor)

                promising = np.zeros(len(sample_actions))
                original = np.zeros((len(sample_actions)))
                for i in range(len(sample_actions)):
                    original[i] = self.__gaussian_density(sample_actions[i].cpu().numpy(), mean=mean_action.detach().cpu().numpy(),
                                                        sigma=self.sigma)
                    promising[i] = self.__gaussian_density(sample_actions[i].cpu().numpy(), mean=mean_action.detach().cpu().numpy(),
                                                        sigma=self.sigma) * np.exp(
                        (1 / self.lamda) * self.phi(normalize_state(torch.Tensor(deepcopy(state)).type(self.FloatTensor)).unsqueeze(0),
                                                    sample_actions[i].unsqueeze(0)).detach().cpu().numpy())
                # promising = np.exp(promising) / np.sum(np.exp(promising))
                # original = np.exp(original) / np.sum(np.exp(original))
                # print(promising)
                promising = softmax(deepcopy(promising))
                original = softmax(deepcopy(original))
                # print(promising)
                action_index = self.__sample_actions_according_to_promising(promising)
                if draw:
                    self.draw_density(sample_actions, original, promising)

                print("online explore actions: %f " % sample_actions[action_index].detach().cpu().numpy())
                return sample_actions[action_index].detach().cpu().numpy()

            else:
                return None
        else:
            print("online explore actions: %f " % mean_action.detach().cpu().numpy())
            return mean_action.detach().cpu().numpy()


    def draw_multiple_explorations(self, state, interact_action=None, draw=True):
        """
        generating exploration actions
        :param interact_action: alternative, if None: use self.u_v to generate base action
        :param state: current state
        :return:
        """
        if interact_action is None and self.u_v is None:
            print("ERROR: No available interaction actions.")
        mean_action = self.u_v.cal(state) if interact_action is None else interact_action
        if self.sampling_way == "weighted":
            sample_actions = np.linspace(0, 10,
                                         self.sampling_num)
            promising_l1_s1 = np.zeros(len(sample_actions))
            promising_l2_s1 = np.zeros(len(sample_actions))
            promising_l1_s2 = np.zeros(len(sample_actions))
            promising_l2_s2 = np.zeros(len(sample_actions))
            original_s1 = np.zeros((len(sample_actions)))
            original_s2 = np.zeros((len(sample_actions)))
            for i in range(len(sample_actions)):
                original_s1[i] = self.__gaussian_density(sample_actions[i], mean=mean_action, sigma=1)
                original_s2[i] = self.__gaussian_density(sample_actions[i], mean=mean_action, sigma=1.1)
                promising_l1_s1[i] = self.__gaussian_density(sample_actions[i], mean=mean_action,
                                                             sigma=1) * np.exp(
                    (1 / 1) * self.phi.cal(state, sample_actions[i]))
                promising_l2_s1[i] = self.__gaussian_density(sample_actions[i], mean=mean_action,
                                                             sigma=1) * np.exp(
                    (1 / 5) * self.phi.cal(state, sample_actions[i]))
                promising_l1_s2[i] = self.__gaussian_density(sample_actions[i], mean=mean_action,
                                                             sigma=1.1) * np.exp(
                    (1 / 1) * self.phi.cal(state, sample_actions[i]))
                promising_l2_s2[i] = self.__gaussian_density(sample_actions[i], mean=mean_action,
                                                             sigma=1.1) * np.exp(
                    (1 / 5) * self.phi.cal(state, sample_actions[i]))
            promising_l1_s1 = np.exp(promising_l1_s1) / np.sum(np.exp(promising_l1_s1))
            promising_l2_s1 = np.exp(promising_l2_s1) / np.sum(np.exp(promising_l2_s1))
            promising_l1_s2 = np.exp(promising_l1_s2) / np.sum(np.exp(promising_l1_s2))
            promising_l2_s2 = np.exp(promising_l2_s2) / np.sum(np.exp(promising_l2_s2))
            original_s1 = np.exp(original_s1) / np.sum(np.exp(original_s1))
            original_s2 = np.exp(original_s2) / np.sum(np.exp(original_s2))
            action_index = self.__sample_actions_according_to_promising(promising_l1_s1)
            if draw:
                sns.set_theme(style="darkgrid")
                l5, = plt.plot(sample_actions, original_s1, marker='D', c='cornflowerblue', lw=2, markersize=0,
                               linestyle="--",
                               label='$\pi_{e,N}$ with $\sigma=1$')
                l6, = plt.plot(sample_actions, original_s2, marker='D', c='tomato', lw=2, markersize=0,
                               linestyle="--",
                               label='$\pi_{e,N}$ with $\sigma=1.1$')

                l1, = plt.plot(sample_actions, promising_l1_s1, marker='D', c='navy', lw=2, markersize=0,
                               linestyle="-",
                               label='$\pi_e$ with $\lambda=1,\sigma=1$')
                l2, = plt.plot(sample_actions, promising_l2_s1, marker='.', c='blue', lw=2, markersize=0,
                               linestyle="-.",
                               label='$\pi_e$ with $\lambda=5,\sigma=1$')
                l3, = plt.plot(sample_actions, promising_l1_s2, marker='D', c='deeppink', lw=2, markersize=0,
                               label='$\pi_e$ with $\lambda=1,\sigma=1.1$')
                l4, = plt.plot(sample_actions, promising_l2_s2, marker='D', c='red', lw=2, markersize=0,
                               linestyle="-.",
                               label='$\pi_e$ with $\lambda=5,\sigma=1.1$')

                plt.legend(loc="best")
                plt.savefig("explore_policy.png", dpi=500)
                plt.show()

            return sample_actions[action_index]

        else:
            return None

    def draw_density(self, x, y_original, y_shift):
        sns.set_theme(style="darkgrid")
        l2, = plt.plot(x, y_original, marker='D', c='cornflowerblue', lw=2, label='original u_v')
        l1, = plt.plot(x, y_shift, marker='D', c='orangered', lw=2, label="explore policy")
        plt.legend(loc="upper right")
        plt.savefig("explore_policy.png", dpi=400)
        plt.show()

    def __sample_actions_according_to_promising(self, promising):
        uni_random = np.random.rand()
        # print(promising)
        cum = 0
        for i in range(self.sampling_num):
            if uni_random < promising[i] + cum:
                print(i)
                return i
            cum += promising[i]
        return self.sampling_num - 1

    def __gaussian_density(self, inp, mean, sigma=None):
        """
        function of calculating gaussian density
        :param inp:
        :param mean:
        :param sigma:
        :return:
        """
        if mean is None:
            print("ERROR: Incorrect mean")
            return None
        sigma = self.sigma if sigma is None else sigma
        return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-(inp - mean) ** 2 / (2 * sigma ** 2))

    def update_lambda(self):
        """
        function of updating lambda
        :return:
        """
        pass


@gin.configurable
class Phi:
    def __init__(self):
        self.alpha = 0.2

    def cal(self, state, action):
        return -self.alpha * (action - 10) ** 2 + 4.2 + 0.02 * np.random.normal()


@gin.configurable
class Uv:
    def __init__(self):
        self.alpha = 1

    def cal(self, state):
        return 4.5


if __name__ == "__main__":
    onlineExplorationPolicy = OnlineExplorationPolicy(phi=Phi, u_v=Uv, )
    # onlineExplorationPolicy.explore_action(state=np.array([1, 2, 3]))
    onlineExplorationPolicy.draw_multiple_explorations(state=np.array([1, 2, 3]))
