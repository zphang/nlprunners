from nlpr.shared.train_setup import TrainSchedule
import nlpr.proj.simple.runner as simple_runner
import nlpr.proj.llp.runner as llp_runner
import nlpr.proj.uda.runner as uda_runner


class UDALLPRunner:
    def __init__(self, task, model_wrapper, optimizer_scheduler, loss_criterion,
                 device,
                 rparams: simple_runner.RunnerParameters,
                 uda_params: uda_runner.UDAParameters,
                 llp_params: llp_runner.LlpParameters,
                 train_schedule: TrainSchedule):
        self.task = task
        self.model_wrapper = model_wrapper
        self.optimizer_scheduler = optimizer_scheduler
        self.loss_criterion = loss_criterion
        self.device = device
        self.rparams = rparams
        self.uda_params = uda_params
        self.llp_params = llp_params
        self.train_schedule = train_schedule

        self.llp_state = None

        # Convenience
        self.model = self.model_wrapper.model

    # LLP
    init_llp_state = llp_runner.LLPRunner.init_llp_state
    create_empty_llp_state = llp_runner.LLPRunner.create_empty_llp_state
    populate_llp_state = llp_runner.LLPRunner.populate_llp_state
    compute_representation_loss = llp_runner.LLPRunner.compute_representation_loss
    run_label_propagate = llp_runner.LLPRunner.run_label_propagate

    # Eval
    run_val = simple_runner.SimpleTaskRunner.run_val
    run_test = simple_runner.SimpleTaskRunner.run_test
    get_eval_dataloader = simple_runner.SimpleTaskRunner.get_eval_dataloader
    complex_propagate = simple_runner.SimpleTaskRunner.complex_backpropagate
