import torch
from torch.utils.data import DataLoader

from aggregation.fed_df import fed_df


from utils.data import filter_dataset


class DataConsumer:
    def __init__(
        self,
        id,
        score_metric,
        model,
        classes_of_interest,
        valset,  # The valset represents the target domain of the DC, if coi=None
        full_testset,
        aggregation_method,  # A method from aggregation.py
        public_dataset,
        batch_size,
        training_epochs,
        fed_prox_mu,
        is_in_competition,  # Determines whether the DC competes for some DOs with another DC
        device,
        true_label_mapping=None,
    ):
        self.id = id
        self.fed_prox_mu = fed_prox_mu
        self.score_metric = score_metric
        self.model = model
        self.val_performance = (0.0, 1000.0)
        self.classes_of_interest = classes_of_interest
        self.valset = valset
        self.testset = filter_dataset(full_testset, self.classes_of_interest)
        self.aggregation_method = aggregation_method
        self.public_dataset = public_dataset
        self.batch_size = batch_size
        self.training_epochs = training_epochs
        self.is_in_competition = is_in_competition
        self.device = device
        self.true_label_mapping = (
            true_label_mapping
            if not true_label_mapping is None
            else torch.tensor(self.classes_of_interest).to(device)
        )

        self.val_performance = self.central_eval(is_test=False)
        self.val_performances = [self.val_performance]
        self.test_performances = [self.central_eval()]

        val_set_size = len(valset)
        print(
            f"DC {self.id}, val_set_size={val_set_size}, COIs={self.classes_of_interest}, M={self.true_label_mapping}, Initial Performance: {self.val_performance}"
        )

    def aggregate_fit(self, results):
        return self.aggregation_method(
            results, self.public_dataset, self.batch_size, self.device
        )

    # model: Model that should be evaluated. When None, the DC's current model is used
    # test: Test set or validation set
    def central_eval(self, model=None, is_test=True):
        data = self.testset if is_test else self.valset
        testloader = DataLoader(data, batch_size=self.batch_size)

        model_to_test = self.model if model is None else model
        acc, loss = model_to_test.test_model(testloader, self.true_label_mapping)

        return acc, loss


    # Evaluates contribution of a DO based on leave-on-out loss
    # FL_Results: List((do_id, model, n_samples))
    def evaluate_contribution(self, fl_results, base_results, do_id):
        m_loo = self.aggregate_fit(
            [(m, n) for (d_id, m, n) in fl_results if not d_id == do_id],
        )
        base_accuracy, base_loss = base_results
        loo_accuracy, loo_loss = self.central_eval(model=m_loo, is_test=False)
        if self.score_metric == "loss" or self.score_metric == "contrloss":
            # Use loss for contribution evaluation
            return loo_loss - base_loss
        else:
            return base_accuracy - loo_accuracy

    # all_do_results: List((do_id, model, n_samples))
    # dos_to_evaluate: List of all DOs whose contribution should be evaluates
    # Returns: Dictionary mapping do_id to contribution for all DOs in dos_to_evaluate
    def evaluate_contributions(self, all_do_results, dos_to_evaluate):
        print()
        m_base = self.aggregate_fit(
            [(m, n) for (_, m, n) in all_do_results],
        )
        base_results = self.central_eval(model=m_base, is_test=False)
        contributions = {}
        for do in dos_to_evaluate:
            do_id = do.id
            contr = self.evaluate_contribution(all_do_results, base_results, do_id)
            print(f"Contribution(DO{do_id}) = {contr}")
            contributions[do_id] = contr
        print()
        return contributions

    # Recursively filters training results until the contributions of all unique DOs are positive
    # A unique DO is a DO in which only this DC is interested
    def filter_training_results(self, training_results, dos_unique):
        if not dos_unique:
            return training_results

        unique_dos_contributions = self.evaluate_contributions(
            training_results, dos_unique
        )
        unique_dos_negative_contributions = {
            do_id: contr
            for do_id, contr in unique_dos_contributions.items()
            if contr < 0
        }
        # If no  DO has negative contribution, no filtering is required
        if len(unique_dos_negative_contributions) == 0:
            return training_results
        # If one  DO has negative contribution, basic filtering is required
        elif len(unique_dos_negative_contributions) == 1:
            print(f"Filtering out DO{list(unique_dos_contributions.keys())[0]}")
            return [
                (do_id, m, n)
                for (do_id, m, n) in training_results
                if not do_id in unique_dos_negative_contributions
            ]
        # Otherwise, remove DO with most negative contribution and refilter recursively
        worst_do_id = min(
            unique_dos_negative_contributions, key=unique_dos_negative_contributions.get
        )
        print(f"Filtering out DO{worst_do_id}")
        # Recursion
        return self.filter_training_results(
            [
                (do_id, m, n)
                for (do_id, m, n) in training_results
                if not do_id == worst_do_id
            ],
            [do for do in dos_unique if not do.id == worst_do_id],
        )

    # Recruited DOs: (list_of_recruited_shared_dos, list_of_recruite_unique_dos)
    def fl_round(self, recruited_dos):
        print(f"TRAINING: DC {self.id}")
        dos_unique = recruited_dos[0]
        dos_shared = recruited_dos[1]
        # Get DO training results
        training_results = [
            (
                do.id,
                *do.fit(
                    self.model,
                    self.true_label_mapping,
                    self.batch_size,
                    self.training_epochs,
                    self.fed_prox_mu,
                ),
            )
            for do in dos_unique + dos_shared
        ]
        # If DC is in a competitive setting, we now evaluate the contributions of the
        # unique DOs that are not competed for to prevent bias

        filtered_training_results = training_results
        if (self.score_metric == "accuracy" or self.score_metric == "greedy_acc") and self.is_in_competition and self.aggregation_method != fed_df:
            filtered_training_results = self.filter_training_results(
                filtered_training_results,
                dos_unique,
            )
        # Calculate new model
        new_model = self.aggregate_fit(
            [(m, n) for (_, m, n) in filtered_training_results]
        )
        if not new_model is None:
            new_val_acc, new_val_loss = self.central_eval(new_model, is_test=False)
            self.val_performances.append((new_val_acc, new_val_loss))
            # Check whether new model should be used based on validation losses
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
                self.model = new_model
                self.test_performances.append((self.central_eval(new_model)))

            else:
                self.test_performances.append(self.test_performances[-1])
        else:
            print("Error: No new Model created!")
