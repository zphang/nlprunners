from nlpr.shared.train_setup import TrainSchedule
import nlpr.proj.llp.runner as llp_runner
import nlpr.proj.uda.runner as uda_runner


class UdaLlpRunner(llp_runner.LLPRunner):
    def __init__(self, task, model_wrapper, optimizer_scheduler, loss_criterion,
                 device,
                 rparams: llp_runner.RunnerParameters,
                 llp_params: llp_runner.LlpParameters,
                 uda_params: uda_runner.UDAParameters,
                 train_schedule: TrainSchedule):
        super().__init__(
            task=task,
            model_wrapper=model_wrapper,
            optimizer_scheduler=optimizer_scheduler,
            loss_criterion=loss_criterion,
            device=device,
            rparams=rparams,
            llp_params=llp_params,
            train_schedule=train_schedule,
        )
        self.uda_params = uda_params

    def run_train_step(self, step, batch, batch_metadata, train_epoch_state):
        batch = batch.to(self.device)
        loss, loss_details = self.compute_representation_loss(batch, batch_metadata)
        loss = self.complex_backpropagate(loss)

        train_epoch_state.tr_loss += loss.item()
        train_epoch_state.nb_tr_examples += len(batch)
        train_epoch_state.nb_tr_steps += 1
        if (step + 1) % self.train_schedule.gradient_accumulation_steps == 0:
            self.optimizer_scheduler.step()
            self.model.zero_grad()
            train_epoch_state.global_step += 1

        # Update memory bank
        with torch.no_grad():
            new_embedding = self.model.forward_batch(batch).embedding
        self.llp_state.big_m_tensor[batch_metadata["example_id"]] = (
                (1 - self.llp_params.llp_mem_bank_t)
                * self.llp_state.big_m_tensor[batch_metadata["example_id"]]
                + self.llp_params.llp_mem_bank_t * new_embedding
        )
        return loss_details
