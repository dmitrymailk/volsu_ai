import torch
import wandb
import datetime
import shutil
import os
from importlib.machinery import SourceFileLoader


class BaseExperiment:
    def __init__(
        self,
        model=None,
        dataloader_train=None,
        dataloader_valid=None,
        dataloader_test=None,
        loss_func_class=None,
        estimate_func_class=None,
        experiment_config=None,
        optimizer_class=None,
        sheduler_class=None,
        project_name=None,
        notebook_name=None,
        name_run="",
        model_description="",
    ):
        assert (
            notebook_name != None
        ), f"notebook_name should be valid filename, but get {notebook_name}"

        # datasets
        self.dataloader_train = dataloader_train
        self.dataloader_valid = dataloader_valid
        self.dataloader_test = dataloader_test

        # wandb
        self.notebook_name = notebook_name
        self.project_name = project_name
        self.experiment_config = experiment_config
        self.wandb_run = None
        self.name_run = name_run
        self.model_description = model_description
        self.model_name = "pytorch_model"
        self.model_artifact = None

        self.optimizer_class = optimizer_class
        self.sheduler_class = sheduler_class
        self.loss_func_class = loss_func_class
        self.estimate_func_class = estimate_func_class

        self.model = model
        self.optimizer = None
        self.sheduler = None
        self.loss_func = None
        self.estimate_func = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device {self.device}")

        # prepare for experiment
        self.setup()
        self.unit_tests()

    def setup(self):
        self.model.to(self.device)
        self.optimizer = self.optimizer_class(
            self.model.parameters(), **self.experiment_config["optimizer"]
        )

        if self.sheduler_class != None:
            self.sheduler = self.sheduler_class(
                self.optimizer, **self.experiment_config["sheduler"]
            )

        self.loss_func = self.loss_func_class()
        self.estimate_func = self.estimate_func_class()

        # set model name
        date_time = self.get_date()
        self.model_name = f"{self.name_run}---{date_time}.pt"
        self.experiment_config["model_name"] = self.model_name

        # setup wandb
        # save model structure and weights to wandb
        self.model_artifact = wandb.Artifact(
            self.name_run,
            type="model",
            description=self.model_description,
            metadata=self.experiment_config,
        )

    def get_date(self):
        now = datetime.now()
        date_time = now.strftime("%m_%d_%Y__%H:%M:%S")
        return date_time

    def unit_tests(self):
        # test training
        X, y = next(iter(self.dataloader_train))
        X, y = X.to(self.device), y.to(self.device)
        pred = self.model(X)
        loss = self.loss_func(pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # test valid
        X, y = next(iter(self.dataloader_valid))
        X, y = X.to(self.device), y.to(self.device)
        pred = self.model(X)
        test_loss = self.estimate_func(pred, y).item()
        correct = (pred.argmax(1) == y).type(torch.float).sum().item()

        # initial validation
        self.model.eval()
        test_loss, correct = 0, 0
        num_batches = len(self.dataloader_valid)
        size = len(self.dataloader_valid.dataset)

        with torch.no_grad():
            for X, y in self.dataloader_valid:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                test_loss += self.estimate_func(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print("Initial val = ", correct)

        print("tests ok")

    def train(self):
        # https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-artifacts/Pipeline_Versioning_with_W%26B_Artifacts.ipynb#scrollTo=qrAWbBV1rd4I
        # если попытаться создать переменную чтобы не городить тут код возникает ошибка с wandb!
        with wandb.init(
            project=self.project_name,
            entity="dimweb",
            settings=wandb.Settings(
                start_method="thread",
                # symlink=False
            ),
            reinit=True,
            name=self.name_run,
            config=self.experiment_config,
            # sync_tensorboard=True
        ) as run:

            self.run = run

            # save model class
            self.save_model_class()

            # start train
            epochs = self.experiment_config["epochs"]
            for i in range(epochs):
                print(f"Epoch: {i}")
                self.train_steps()
                self.valid_steps()

            # sync model
            self.wandb_save_model()

            print(f"train end")

    def save_model_class(self):
        # save class
        model_class_name = self.experiment_config["model_class_name"]
        class_script_path_dest = f"{os.path.join(wandb.run.dir, model_class_name)}.py"
        class_script_path_src = f"./models/{model_class_name}.py"
        shutil.copy2(class_script_path_src, class_script_path_dest)
        self.model_artifact.add_file(class_script_path_dest)
        wandb.save(class_script_path_dest)

    def wandb_save_model(self):
        # wandb использует symlinks для того чтобы сохранять файлы
        # но из-за проблем с правами доступа возникает ошибка и модель нельзя сохранить
        # поэтому пришлось сохранять модель в дирректорию с самим запуском
        # https://docs.wandb.ai/guides/track/advanced/save-restore#example-of-saving-a-file-to-the-wandb-run-directory
        model_save_path = os.path.join(wandb.run.dir, self.model_name)
        torch.save(self.model.state_dict(), model_save_path)
        self.model_artifact.add_file(model_save_path)
        wandb.save(model_save_path)

        # save notebook
        notebook_path = os.path.join(wandb.run.dir, self.notebook_name)
        shutil.copy2(self.notebook_name, notebook_path)
        self.model_artifact.add_file(notebook_path)
        wandb.save(notebook_path)

        wandb.log_artifact(self.model_artifact)

    def train_steps(self):
        raise NotImplementedError("You need specify training steps")

    def valid_steps(self):
        raise NotImplementedError("You need specify valid steps")

    def load_model(self, artifact_name, additional_model_args={}):
        assert artifact_name != ""
        with wandb.init(project=self.project_name, job_type="inference"):
            model_artifact = wandb.use_artifact(artifact_name)
            model_dir = model_artifact.download()
            model_config = model_artifact.metadata
            model_path = os.path.join(model_dir, model_config["model_name"])
            # print(model_config)

            model_class_name = model_config["model_class_name"]
            model_script_path = f"./artifacts/{artifact_name}/{model_class_name}.py"
            # get module by path https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path?rq=1
            model_class = getattr(
                SourceFileLoader(model_class_name, model_script_path).load_module(),
                model_class_name,
            )

            model_args = model_config["model_args"]
            model = model_class(**model_args, **additional_model_args)

            model.load_state_dict(torch.load(model_path))
            self.model = model
            self.model.to(self.device)

    @staticmethod
    def static_load_model(artifact_name="", project_name="", additional_model_args={}):
        assert artifact_name != ""
        assert project_name != ""
        with wandb.init(project=project_name, job_type="inference"):
            model_artifact = wandb.use_artifact(artifact_name)
            model_dir = model_artifact.download()
            model_config = model_artifact.metadata
            model_path = os.path.join(model_dir, model_config["model_name"])

            model_class_name = model_config["model_class_name"]
            model_script_path = f"./artifacts/{artifact_name}/{model_class_name}.py"
            model_class = getattr(
                SourceFileLoader(model_class_name, model_script_path).load_module(),
                model_class_name,
            )

            model_args = model_config["model_args"]
            model = model_class(**model_args, **additional_model_args)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.load_state_dict(torch.load(model_path))

            return model

    def test(self, artifact_name="", model=None):
        raise NotImplementedError("You need specify test steps")


class Experiment(BaseExperiment):
    def __init__(self, **kwargs):
        super(Experiment, self).__init__(**kwargs)

    def train_steps(self):
        self.model.train()
        interval = self.experiment_config["check_interval"]

        for batch, (X, y) in enumerate(self.dataloader_train):
            # Send data to training device
            X, y = X.to(self.device), y.to(self.device)

            # Compute prediction error
            pred = self.model(X)
            loss = self.loss_func(pred, y)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.sheduler != None:
                self.sheduler.step()

            # Progress output
            if batch % interval == 0:
                wandb.log({"train_loss": loss.item()})

    def valid_steps(self):
        self.model.eval()
        test_loss, correct = 0, 0
        num_batches = len(self.dataloader_valid)
        size = len(self.dataloader_valid.dataset)

        with torch.no_grad():
            for X, y in self.dataloader_valid:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                test_loss += self.estimate_func(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size

        wandb.log({"val_loss": test_loss})
        wandb.log({"val_acc": correct})

    def test(self, artifact_name="", model=None):
        if model is None:
            self.load_model(artifact_name)
        else:
            self.model = model
            self.model.to(self.device)

        print("model loaded to disk")
        predictions = []

        self.model.eval()

        with torch.no_grad():
            for X, _ in self.test_dataloader:
                X = X.to(self.device)
                pred = self.model(X).argmax(1).cpu().numpy()
                predictions.extend(list(pred))

        date_time = self.get_date()
        filename = f"./predictions/{self.name_run}---{date_time}.csv"
        with open(filename, "w") as solution:
            print("Id,Category", file=solution)
            for i, label in enumerate(predictions):
                print(f"{i},{label}", file=solution)
        print("test end")
