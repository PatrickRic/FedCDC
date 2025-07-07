import copy

import torch

from torch.utils.data import DataLoader

from utils.data import filter_dataset
from aggregation.ensemble_distillation.ekd import (
    perform_ensemble_distillation,
    get_teacher_to_student_mappings,
)


class SuperDataConsumer:
    def __init__(
        self,
        id,
        score_metric,
        expert_dcs,  # List of DCs whose models will be distilled into the  SuperDC model
        model,
        classes_of_interest,
        valset,  # The valset that represents the target domain of the DC
        full_testset,
        public_dataset,
        init_test_performances,
        init_val_performances,
        training_epochs,
        batch_size,
        learning_rate,
        temperature,
        hard_loss_weight,
        device,
    ):
        self.score_metric = score_metric
        self.model = model
        self.id = id
        self.test_performances = init_test_performances
        self.classes_of_interest = classes_of_interest
        self.valset = valset
        self.testset = filter_dataset(full_testset, self.classes_of_interest)
        self.n_shared_dos_used = 0
        self.batch_size = batch_size
        self.training_epochs = training_epochs
        self.device = device
        self.public_dataset = public_dataset
        self.expert_dcs = expert_dcs
        self.lr = learning_rate
        self.temperature = temperature
        self.hard_loss_weight = hard_loss_weight
        #self.selection_frequency = selection_frequency
        #self.selection_ratio = selection_ratio
        self.true_label_mapping = torch.tensor(classes_of_interest).to(device)

        self.val_performance = self.central_eval(is_test=False)
        self.val_performances = init_val_performances[:-1] + [self.val_performance]

        val_set_size = len(valset)

        # Determine teacher to student output mappings
        self.teacher_to_student_mappings = get_teacher_to_student_mappings(
            expert_dcs, self.true_label_mapping
        )

        expert_dc_descriptions = [f"DC{dc.id}" for dc in expert_dcs]

        print(
            f"SUPER-DC {self.id}, val_set_size={val_set_size}, COIs={self.classes_of_interest}, M={self.true_label_mapping}, , Expert DCs: {expert_dc_descriptions}"
        )

        print()

    # model: Model that should be evaluated. When None, the DC's current model is used
    # test: Test set or validation set
    def central_eval(self, model=None, is_test=True):
        data = self.testset if is_test else self.valset
        testloader = DataLoader(data, batch_size=self.batch_size)

        m = self.model if model == None else model
        acc, loss = m.test_model(testloader, self.true_label_mapping)

        return acc, loss

    def fl_round(self):
        print(f"TRAINING: SUPER-DC {self.id}")

        # Ensemble Distillation
        teacher_models = [dc.model for dc in self.expert_dcs]
        student_model = copy.deepcopy(self.model).to(self.device)
        distillation_loader = DataLoader(
            self.public_dataset, batch_size=self.batch_size, shuffle=True
        )
        # Perform EKD
        perform_ensemble_distillation(
            teacher_models,
            student_model,
            self.teacher_to_student_mappings,
            distillation_loader,
            # self.public_dataset,
            # self.batch_size,
            self.training_epochs,
            self.lr,
            self.temperature,
            self.hard_loss_weight,
            # self.selection_frequency,
            # self.selection_ratio,
            self.device,
        )

        new_val_acc, new_val_loss = self.central_eval(
            model=student_model, is_test=False
        )
        self.val_performances.append((new_val_acc, new_val_loss))
        # Check whether new model should be used based on validation accuracies
        if self.score_metric == "loss":
            new_is_better = new_val_loss <= self.val_performance[1]
        elif self.score_metric == "greedy_acc":
            new_is_better = new_val_acc >= self.val_performance[0]
        else:
            new_is_better = True
        print(
            f"New Model (Val Acc, Val Loss) = ({new_val_acc}, {new_val_loss}) --> New model used: {new_is_better}"
        )
        print()
        if new_is_better:
            self.val_performance = (new_val_acc, new_val_loss)
            self.model = student_model
            self.test_performances.append((self.central_eval(student_model)))
        else:
            self.test_performances.append(self.test_performances[-1])
        """# Give new model to own expert DC
        for dc in self.expert_dcs:
            if dc.id != "Alliance":
                print(f"Transferring new model to expert: {dc.id}")
                dc.model = self.model"""
