import os
import sys
import numpy as np
from tqdm import tqdm
import tensorflow as tf

sys.path.append("./planner_model")

from data_loader import PlanDataset
from trajectory_model import PlanLearner
from RWTA_Loss import RWTALoss, TrajectoryCostLoss

class TrajectoryTrainer(object):
    def __init__(self):

        self.network = PlanLearner()
        self.space_loss = RWTALoss()
        self.cost_loss = TrajectoryCostLoss()
        self.cost_loss_v = TrajectoryCostLoss()

        self.min_val_loss = tf.Variable(np.inf, name="min_val_loss", trainable=False)

        # scheduler
        self.learning_rate_fn = tf.keras.experimental.CosineDecayRestarts(
            1e-3, 50000, 1.5, 0.75, 0.01
        )

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate_fn)

        self.train_space_loss = tf.keras.metrics.Mean(name="train_space_loss")
        self.val_space_loss = tf.keras.metrics.Mean(name="validation_space_loss")
        self.train_cost_loss = tf.keras.metrics.Mean(name="train_cost_loss")
        self.val_cost_loss = tf.keras.metrics.Mean(name="validation_cost_loss")

        self.global_epoch = tf.Variable(0)
        self.ckpt = tf.train.Checkpoint(
            step=self.global_epoch, optimizer=self.optimizer, net=self.network
        )

        print("********************************************")
        print("Initialized trainer.")
        print("********************************************")

    @tf.function
    def train_iter(self, inputs, labels):
        with tf.GradientTape() as tape:
            predictions = self.network(inputs)
            space_loss = self.space_loss(labels, predictions)
            cost_loss = self.cost_loss((inputs["env"], inputs["state"]), predictions)
            # print(f"space loss: {space_loss[0]}\tcost loss: {cost_loss[0]}")
            # Equation 7 in paper
            loss = space_loss + cost_loss

        gradients = tape.gradient(loss, self.network.trainable_variables)
        gradients = [tf.clip_by_norm(g, 1) for g in gradients]
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))
        self.train_space_loss.update_state(space_loss)
        self.train_cost_loss.update_state(cost_loss)
        return gradients

    @tf.function
    def val_iter(self, inputs, labels):

        predictions = self.network(inputs)
        space_loss = self.space_loss(labels, predictions)
        cost_loss = self.cost_loss_v((inputs["env"], inputs["state"]), predictions)
        self.val_space_loss.update_state(space_loss)
        self.val_cost_loss.update_state(cost_loss)

        return predictions

    def format_data(self, features):

        inputs = {"state": features[:, :, :16], "env": features[:, :, 16:]}
        return inputs

    def write_train_summaries(self, features, gradients):

        with self.summary_writer.as_default():
            tf.summary.scalar('Train Space Loss', self.train_space_loss.result(),
                              step=self.optimizer.iterations)
            tf.summary.scalar('Train Traj_Cost Loss', self.train_cost_loss.result(),
                              step=self.optimizer.iterations)

    def train(self):

        print("Training...")

        if not hasattr(self, "train_log_dir"):
            self.train_log_dir = os.path.join(os.getcwd(), "train")
            self.summary_writer = tf.summary.create_file_writer(self.train_log_dir)
            self.ckpt_manager = tf.train.CheckpointManager(
                self.ckpt, self.train_log_dir, max_to_keep=20
            )
        else:
            self.min_val_loss = np.inf
            self.train_space_loss.reset_states()
            self.val_space_loss.reset_states()

        dataset_train = PlanDataset("planner_model/data/", training=True)
        dataset_val = PlanDataset("planner_model/data/", training=False)

        for epoch in range(150):
            # Training
            tf.keras.backend.set_learning_phase(1)
            for k, (features, labels) in enumerate(tqdm(dataset_train.dataset)):
                features = self.format_data(features)
                gradients = self.train_iter(features, labels)

            if tf.equal(k % 200, 0):
                self.write_train_summaries(features, gradients)
                self.train_space_loss.reset_states()
                self.train_cost_loss.reset_states()

            # Testing
            tf.keras.backend.set_learning_phase(0)
            for k, (features, labels) in enumerate(tqdm(dataset_val.dataset)):
                features = self.format_data(features)
                self.val_iter(features, labels)
            val_space_loss = self.val_cost_loss.result()
            val_cost_loss = self.val_cost_loss.result()
            validation_loss = val_space_loss + val_cost_loss

            with self.summary_writer.as_default():
                tf.summary.scalar("Validation Space Loss", val_space_loss, step=tf.cast(self.global_epoch, dtype=tf.int64))
                tf.summary.scalar("Validation Cost Loss", val_cost_loss, step=tf.cast(self.global_epoch, dtype=tf.int64))

            self.val_space_loss.reset_states()
            self.val_cost_loss.reset_states()

            self.global_epoch = self.global_epoch + 1
            self.ckpt.step.assign_add(1)

            print(
                "Epoch: {:2d}, Val Space Loss: {:4f}, Val Cost Loss: {:.4f}".format(
                    self.global_epoch, val_space_loss, val_cost_loss
                )
            )

            if validation_loss < self.min_val_loss:
                if validation_loss < self.min_val_loss:
                    self.min_val_loss = validation_loss
                    save_path = self.ckpt_manager.save()
                    print("Saved Checkpoint for epoch {}: {}".format(int(self.ckpt.step), save_path))

        print("------------------------------")
        print("Training finished successfully")
        print("------------------------------")

    def inference(self, inputs):

        processed_pred = self.full_post_inference(inputs).numpy()
        processed_pred = processed_pred[:,np.abs(processed_pred[0, :, 0]).argsort(), :]
        pred_costs = np.abs(processed_pred[0,:,0])
        predictions = processed_pred[0, :, 1:]
        return pred_costs, predictions

    @tf.function
    def full_post_inference(self, inputs):

        predictions = self.network(inputs)
        return predictions

if __name__ == "__main__":
    trainer = TrajectoryTrainer()
    trainer.train()
