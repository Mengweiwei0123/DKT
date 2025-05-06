import time


class ExperimentLogger:
    def __init__(self, config):
        self.epoch_count = 0
        self.start_time = time.time()
        self.train_auc_history = []
        self.test_auc_history = []
        self.best_metrics = {'auc': 0.0, 'accuracy': 0.0}
        self.best_model_state = None
        self.best_epoch = -1
        self.patience = config.patience
        self.run_duration = ''
        self.end_timestamp = ''

    def log_epoch(self, epoch, train_metrics, test_metrics, model):
        log_message = (
            f"**************************************** Epoch-{epoch} ****************************************\n"
            f"Train metrics: AUC={train_metrics['auc']} ACC={train_metrics['acc']} RMSE={train_metrics['rmse']}, Test metrics: AUC={test_metrics['auc']} ACC={test_metrics['acc']} RMSE={test_metrics['rmse']}\n"
        )
        print(log_message.rstrip('\n'))

        if self.best_metrics['auc'] < test_metrics['auc']:
            self.best_metrics = test_metrics.copy()
            self.best_epoch = epoch
            self.best_model_state = model.state_dict()

        self._update_auc_history(train_metrics['auc'], test_metrics['auc'])

    def finalize_run(self, config):
        self._capture_run_time()
        run_summary = {
            'end_time': self.end_timestamp,
            'duration': self.run_duration,
            'total_epochs': self.epoch_count,
            'best_epoch': self.best_epoch
        }
        run_summary.update(self.best_metrics)
        run_summary.update(vars(config))
        print(str(run_summary).rstrip('\n'))

    def _update_auc_history(self, train_auc, test_auc):
        self.train_auc_history.append(train_auc)
        self.test_auc_history.append(test_auc)

    def should_stop_early(self):
        if len(self.test_auc_history) < self.patience:
            return False
        recent_aucs = self.test_auc_history[-self.patience:]
        return max(recent_aucs[1:]) < recent_aucs[0]

    def _capture_run_time(self):
        self.end_timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        elapsed_time = time.time() - self.start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        self.run_duration = f"{int(hours)}:{int(minutes)}:{int(seconds)}"

    def increment_epoch(self):
        self.epoch_count += 1
