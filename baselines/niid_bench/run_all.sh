seed=$1

# iid cifar10
python -m niid_bench.main_fedavg partitioning=iid dataset_seed=$seed &
python -m niid_bench.main_scaffold partitioning=iid dataset_seed=$seed &
python -m niid_bench.main_fedprox partitioning=iid dataset_seed=$seed &
python -m niid_bench.main_fedprox partitioning=iid mu=0.1 dataset_seed=$seed &
python -m niid_bench.main_fedprox partitioning=iid mu=0.001 dataset_seed=$seed &
python -m niid_bench.main_fedprox partitioning=iid mu=1.0 dataset_seed=$seed &
python -m niid_bench.main_fednova partitioning=iid dataset_seed=$seed;

# iid mnist
python -m niid_bench.main_fedavg partitioning=iid dataset_seed=$seed dataset_name=mnist model_t=niid_bench.models.CNNMnist model.input_dim=256 &
python -m niid_bench.main_scaffold partitioning=iid dataset_seed=$seed dataset_name=mnist model_t=niid_bench.models.CNNMnist model.input_dim=256 &
python -m niid_bench.main_fedprox partitioning=iid dataset_seed=$seed dataset_name=mnist model_t=niid_bench.models.CNNMnist model.input_dim=256 &
python -m niid_bench.main_fedprox partitioning=iid mu=0.1 dataset_seed=$seed dataset_name=mnist model_t=niid_bench.models.CNNMnist model.input_dim=256 &
python -m niid_bench.main_fedprox partitioning=iid mu=0.001 dataset_seed=$seed dataset_name=mnist model_t=niid_bench.models.CNNMnist model.input_dim=256 &
python -m niid_bench.main_fedprox partitioning=iid mu=1.0 dataset_seed=$seed dataset_name=mnist model_t=niid_bench.models.CNNMnist model.input_dim=256 &
python -m niid_bench.main_fednova partitioning=iid dataset_seed=$seed dataset_name=mnist model_t=niid_bench.models.CNNMnist model.input_dim=256;

# iid fmnist
python -m niid_bench.main_fedavg partitioning=iid dataset_seed=$seed dataset_name=fmnist model_t=niid_bench.models.CNNMnist model.input_dim=256 &
python -m niid_bench.main_scaffold partitioning=iid dataset_seed=$seed dataset_name=fmnist model_t=niid_bench.models.CNNMnist model.input_dim=256 &
python -m niid_bench.main_fedprox partitioning=iid dataset_seed=$seed dataset_name=fmnist model_t=niid_bench.models.CNNMnist model.input_dim=256 &
python -m niid_bench.main_fedprox partitioning=iid mu=0.1 dataset_seed=$seed dataset_name=fmnist model_t=niid_bench.models.CNNMnist model.input_dim=256 &
python -m niid_bench.main_fedprox partitioning=iid mu=0.001 dataset_seed=$seed dataset_name=fmnist model_t=niid_bench.models.CNNMnist model.input_dim=256 &
python -m niid_bench.main_fedprox partitioning=iid mu=1.0 dataset_seed=$seed dataset_name=fmnist model_t=niid_bench.models.CNNMnist model.input_dim=256 &
python -m niid_bench.main_fednova partitioning=iid dataset_seed=$seed dataset_name=fmnist model_t=niid_bench.models.CNNMnist model.input_dim=256;

# dirichlet cifar10
python -m niid_bench.main_fedavg partitioning=dirichlet dataset_seed=$seed &
python -m niid_bench.main_scaffold partitioning=dirichlet dataset_seed=$seed &
python -m niid_bench.main_fedprox partitioning=dirichlet dataset_seed=$seed &
python -m niid_bench.main_fedprox partitioning=dirichlet mu=0.1 dataset_seed=$seed &
python -m niid_bench.main_fedprox partitioning=dirichlet mu=0.001 dataset_seed=$seed &
python -m niid_bench.main_fedprox partitioning=dirichlet mu=1.0 dataset_seed=$seed &
python -m niid_bench.main_fednova partitioning=dirichlet dataset_seed=$seed;

# dirichlet mnist
python -m niid_bench.main_fedavg partitioning=dirichlet dataset_seed=$seed dataset_name=mnist model_t=niid_bench.models.CNNMnist model.input_dim=256 &
python -m niid_bench.main_scaffold partitioning=dirichlet dataset_seed=$seed dataset_name=mnist model_t=niid_bench.models.CNNMnist model.input_dim=256 &
python -m niid_bench.main_fedprox partitioning=dirichlet dataset_seed=$seed dataset_name=mnist model_t=niid_bench.models.CNNMnist model.input_dim=256 &
python -m niid_bench.main_fedprox partitioning=dirichlet mu=0.1 dataset_seed=$seed dataset_name=mnist model_t=niid_bench.models.CNNMnist model.input_dim=256 &
python -m niid_bench.main_fedprox partitioning=dirichlet mu=0.001 dataset_seed=$seed dataset_name=mnist model_t=niid_bench.models.CNNMnist model.input_dim=256 &
python -m niid_bench.main_fedprox partitioning=dirichlet mu=1.0 dataset_seed=$seed dataset_name=mnist model_t=niid_bench.models.CNNMnist model.input_dim=256 &
python -m niid_bench.main_fednova partitioning=dirichlet dataset_seed=$seed dataset_name=mnist model_t=niid_bench.models.CNNMnist model.input_dim=256;

# dirichlet fmnist
python -m niid_bench.main_fedavg partitioning=dirichlet dataset_seed=$seed dataset_name=fmnist model_t=niid_bench.models.CNNMnist model.input_dim=256 &
python -m niid_bench.main_scaffold partitioning=dirichlet dataset_seed=$seed dataset_name=fmnist model_t=niid_bench.models.CNNMnist model.input_dim=256 &
python -m niid_bench.main_fedprox partitioning=dirichlet dataset_seed=$seed dataset_name=fmnist model_t=niid_bench.models.CNNMnist model.input_dim=256 &
python -m niid_bench.main_fedprox partitioning=dirichlet mu=0.1 dataset_seed=$seed dataset_name=fmnist model_t=niid_bench.models.CNNMnist model.input_dim=256 &
python -m niid_bench.main_fedprox partitioning=dirichlet mu=0.001 dataset_seed=$seed dataset_name=fmnist model_t=niid_bench.models.CNNMnist model.input_dim=256 &
python -m niid_bench.main_fedprox partitioning=dirichlet mu=1.0 dataset_seed=$seed dataset_name=fmnist model_t=niid_bench.models.CNNMnist model.input_dim=256 &
python -m niid_bench.main_fednova partitioning=dirichlet dataset_seed=$seed dataset_name=fmnist model_t=niid_bench.models.CNNMnist model.input_dim=256;

# label_quantity cifar10
python -m niid_bench.main_fedavg partitioning=label_quantity dataset_seed=$seed &
python -m niid_bench.main_scaffold partitioning=label_quantity dataset_seed=$seed &
python -m niid_bench.main_fedprox partitioning=label_quantity dataset_seed=$seed &
python -m niid_bench.main_fedprox partitioning=label_quantity mu=0.1 dataset_seed=$seed &
python -m niid_bench.main_fedprox partitioning=label_quantity mu=0.001 dataset_seed=$seed &
python -m niid_bench.main_fedprox partitioning=label_quantity mu=1.0 dataset_seed=$seed &
python -m niid_bench.main_fednova partitioning=label_quantity dataset_seed=$seed;

# label_quantity mnist
python -m niid_bench.main_fedavg partitioning=label_quantity dataset_seed=$seed dataset_name=mnist model_t=niid_bench.models.CNNMnist model.input_dim=256 &
python -m niid_bench.main_scaffold partitioning=label_quantity dataset_seed=$seed dataset_name=mnist model_t=niid_bench.models.CNNMnist model.input_dim=256 &
python -m niid_bench.main_fedprox partitioning=label_quantity dataset_seed=$seed dataset_name=mnist model_t=niid_bench.models.CNNMnist model.input_dim=256 &
python -m niid_bench.main_fedprox partitioning=label_quantity mu=0.1 dataset_seed=$seed dataset_name=mnist model_t=niid_bench.models.CNNMnist model.input_dim=256 &
python -m niid_bench.main_fedprox partitioning=label_quantity mu=0.001 dataset_seed=$seed dataset_name=mnist model_t=niid_bench.models.CNNMnist model.input_dim=256 &
python -m niid_bench.main_fedprox partitioning=label_quantity mu=1.0 dataset_seed=$seed dataset_name=mnist model_t=niid_bench.models.CNNMnist model.input_dim=256 &
python -m niid_bench.main_fednova partitioning=label_quantity dataset_seed=$seed dataset_name=mnist model_t=niid_bench.models.CNNMnist model.input_dim=256;

# label_quantity fmnist
python -m niid_bench.main_fedavg partitioning=label_quantity dataset_seed=$seed dataset_name=fmnist model_t=niid_bench.models.CNNMnist model.input_dim=256 &
python -m niid_bench.main_scaffold partitioning=label_quantity dataset_seed=$seed dataset_name=fmnist model_t=niid_bench.models.CNNMnist model.input_dim=256 &
python -m niid_bench.main_fedprox partitioning=label_quantity dataset_seed=$seed dataset_name=fmnist model_t=niid_bench.models.CNNMnist model.input_dim=256 &
python -m niid_bench.main_fedprox partitioning=label_quantity mu=0.1 dataset_seed=$seed dataset_name=fmnist model_t=niid_bench.models.CNNMnist model.input_dim=256 &
python -m niid_bench.main_fedprox partitioning=label_quantity mu=0.001 dataset_seed=$seed dataset_name=fmnist model_t=niid_bench.models.CNNMnist model.input_dim=256 &
python -m niid_bench.main_fedprox partitioning=label_quantity mu=1.0 dataset_seed=$seed dataset_name=fmnist model_t=niid_bench.models.CNNMnist model.input_dim=256 &
python -m niid_bench.main_fednova partitioning=label_quantity dataset_seed=$seed dataset_name=fmnist model_t=niid_bench.models.CNNMnist model.input_dim=256;

# label_quantity with 1 label per client cifar10
python -m niid_bench.main_fedavg partitioning=label_quantity dataset_seed=$seed labels_per_client=1 &
python -m niid_bench.main_scaffold partitioning=label_quantity dataset_seed=$seed labels_per_client=1 &
python -m niid_bench.main_fedprox partitioning=label_quantity dataset_seed=$seed labels_per_client=1 &
python -m niid_bench.main_fedprox partitioning=label_quantity mu=0.1 dataset_seed=$seed labels_per_client=1 &
python -m niid_bench.main_fedprox partitioning=label_quantity mu=0.001 dataset_seed=$seed labels_per_client=1 &
python -m niid_bench.main_fedprox partitioning=label_quantity mu=1.0 dataset_seed=$seed labels_per_client=1 &
python -m niid_bench.main_fednova partitioning=label_quantity dataset_seed=$seed labels_per_client=1;

# label_quantity with 1 label per client mnist
python -m niid_bench.main_fedavg partitioning=label_quantity dataset_seed=$seed dataset_name=mnist model_t=niid_bench.models.CNNMnist model.input_dim=256 labels_per_client=1 &
python -m niid_bench.main_scaffold partitioning=label_quantity dataset_seed=$seed dataset_name=mnist model_t=niid_bench.models.CNNMnist model.input_dim=256 labels_per_client=1 &
python -m niid_bench.main_fedprox partitioning=label_quantity dataset_seed=$seed dataset_name=mnist model_t=niid_bench.models.CNNMnist model.input_dim=256 labels_per_client=1 &
python -m niid_bench.main_fedprox partitioning=label_quantity mu=0.1 dataset_seed=$seed dataset_name=mnist model_t=niid_bench.models.CNNMnist model.input_dim=256 labels_per_client=1 &
python -m niid_bench.main_fedprox partitioning=label_quantity mu=0.001 dataset_seed=$seed dataset_name=mnist model_t=niid_bench.models.CNNMnist model.input_dim=256 labels_per_client=1 &
python -m niid_bench.main_fedprox partitioning=label_quantity mu=1.0 dataset_seed=$seed dataset_name=mnist model_t=niid_bench.models.CNNMnist model.input_dim=256 labels_per_client=1 &
python -m niid_bench.main_fednova partitioning=label_quantity dataset_seed=$seed dataset_name=mnist model_t=niid_bench.models.CNNMnist model.input_dim=256 labels_per_client=1;

# label_quantity with 1 label per client fmnist
python -m niid_bench.main_fedavg partitioning=label_quantity dataset_seed=$seed dataset_name=fmnist model_t=niid_bench.models.CNNMnist model.input_dim=256 labels_per_client=1 &
python -m niid_bench.main_scaffold partitioning=label_quantity dataset_seed=$seed dataset_name=fmnist model_t=niid_bench.models.CNNMnist model.input_dim=256 labels_per_client=1 &
python -m niid_bench.main_fedprox partitioning=label_quantity dataset_seed=$seed dataset_name=fmnist model_t=niid_bench.models.CNNMnist model.input_dim=256 labels_per_client=1 &
python -m niid_bench.main_fedprox partitioning=label_quantity mu=0.1 dataset_seed=$seed dataset_name=fmnist model_t=niid_bench.models.CNNMnist model.input_dim=256 labels_per_client=1 &
python -m niid_bench.main_fedprox partitioning=label_quantity mu=0.001 dataset_seed=$seed dataset_name=fmnist model_t=niid_bench.models.CNNMnist model.input_dim=256 labels_per_client=1 &
python -m niid_bench.main_fedprox partitioning=label_quantity mu=1.0 dataset_seed=$seed dataset_name=fmnist model_t=niid_bench.models.CNNMnist model.input_dim=256 labels_per_client=1 &
python -m niid_bench.main_fednova partitioning=label_quantity dataset_seed=$seed dataset_name=fmnist model_t=niid_bench.models.CNNMnist model.input_dim=256 labels_per_client=1;

# label_quantity with 3 labels per client cifar10
python -m niid_bench.main_fedavg partitioning=label_quantity dataset_seed=$seed labels_per_client=3 &
python -m niid_bench.main_scaffold partitioning=label_quantity dataset_seed=$seed labels_per_client=3 &
python -m niid_bench.main_fedprox partitioning=label_quantity dataset_seed=$seed labels_per_client=3 &
python -m niid_bench.main_fedprox partitioning=label_quantity mu=0.1 dataset_seed=$seed labels_per_client=3 &
python -m niid_bench.main_fedprox partitioning=label_quantity mu=0.001 dataset_seed=$seed labels_per_client=3 &
python -m niid_bench.main_fedprox partitioning=label_quantity mu=1.0 dataset_seed=$seed labels_per_client=3 &
python -m niid_bench.main_fednova partitioning=label_quantity dataset_seed=$seed labels_per_client=3;

# label_quantity with 3 labels per client mnist
python -m niid_bench.main_fedavg partitioning=label_quantity dataset_seed=$seed dataset_name=mnist model_t=niid_bench.models.CNNMnist model.input_dim=256 labels_per_client=3 &
python -m niid_bench.main_scaffold partitioning=label_quantity dataset_seed=$seed dataset_name=mnist model_t=niid_bench.models.CNNMnist model.input_dim=256 labels_per_client=3 &
python -m niid_bench.main_fedprox partitioning=label_quantity dataset_seed=$seed dataset_name=mnist model_t=niid_bench.models.CNNMnist model.input_dim=256 labels_per_client=3 &
python -m niid_bench.main_fedprox partitioning=label_quantity mu=0.1 dataset_seed=$seed dataset_name=mnist model_t=niid_bench.models.CNNMnist model.input_dim=256 labels_per_client=3 &
python -m niid_bench.main_fedprox partitioning=label_quantity mu=0.001 dataset_seed=$seed dataset_name=mnist model_t=niid_bench.models.CNNMnist model.input_dim=256 labels_per_client=3 &
python -m niid_bench.main_fedprox partitioning=label_quantity mu=1.0 dataset_seed=$seed dataset_name=mnist model_t=niid_bench.models.CNNMnist model.input_dim=256 labels_per_client=3 &
python -m niid_bench.main_fednova partitioning=label_quantity dataset_seed=$seed dataset_name=mnist model_t=niid_bench.models.CNNMnist model.input_dim=256 labels_per_client=3;

# label_quantity with 3 labels per client fmnist
python -m niid_bench.main_fedavg partitioning=label_quantity dataset_seed=$seed dataset_name=fmnist model_t=niid_bench.models.CNNMnist model.input_dim=256 labels_per_client=3 &
python -m niid_bench.main_scaffold partitioning=label_quantity dataset_seed=$seed dataset_name=fmnist model_t=niid_bench.models.CNNMnist model.input_dim=256 labels_per_client=3 &
python -m niid_bench.main_fedprox partitioning=label_quantity dataset_seed=$seed dataset_name=fmnist model_t=niid_bench.models.CNNMnist model.input_dim=256 labels_per_client=3 &
python -m niid_bench.main_fedprox partitioning=label_quantity mu=0.1 dataset_seed=$seed dataset_name=fmnist model_t=niid_bench.models.CNNMnist model.input_dim=256 labels_per_client=3 &
python -m niid_bench.main_fedprox partitioning=label_quantity mu=0.001 dataset_seed=$seed dataset_name=fmnist model_t=niid_bench.models.CNNMnist model.input_dim=256 labels_per_client=3 &
python -m niid_bench.main_fedprox partitioning=label_quantity mu=1.0 dataset_seed=$seed dataset_name=fmnist model_t=niid_bench.models.CNNMnist model.input_dim=256 labels_per_client=3 &
python -m niid_bench.main_fednova partitioning=label_quantity dataset_seed=$seed dataset_name=fmnist model_t=niid_bench.models.CNNMnist model.input_dim=256 labels_per_client=3;
