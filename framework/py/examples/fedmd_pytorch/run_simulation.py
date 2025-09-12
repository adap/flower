import torch
import numpy as np
from collections import defaultdict

from examples.fedmd_pytorch.server import build_server_and_strategy
from examples.fedmd_pytorch.client import make_client
from flwr.server.app_fedmd import run_fedmd_training
from flwr.server.client_manager import ClientManager

class FedMDValidator:
    """FedMD 검증을 위한 클래스"""
    def __init__(self, clients):
        self.clients = clients
        self.round_logits = defaultdict(list)
        self.round_losses = defaultdict(list)
        self.round_accuracies = defaultdict(list)
        
    def evaluate_client_models(self, round_num):
        """각 클라이언트 모델의 성능 평가"""
        print(f"\n=== Round {round_num} - Client Model Evaluation ===")
        
        for i, client_proxy in enumerate(self.clients):
            client = client_proxy.client
            model = client.model
            device = client.device
            
            # 테스트 데이터로 평가
            test_loader = self._get_test_loader()
            accuracy, loss = self._evaluate_model(model, test_loader, device)
            
            self.round_accuracies[round_num].append(accuracy)
            self.round_losses[round_num].append(loss)
            
            print(f"Client {i+1}: Accuracy = {accuracy:.4f}, Loss = {loss:.4f}")
    
    def _get_test_loader(self):
        """테스트 데이터 로더 생성"""
        import torchvision
        import torchvision.transforms as T
        from torch.utils.data import DataLoader
        
        transform = T.Compose([T.ToTensor()])
        test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
        return DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    def _evaluate_model(self, model, test_loader, device):
        """모델 평가"""
        model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        
        criterion = torch.nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = correct / total
        avg_loss = total_loss / len(test_loader)
        return accuracy, avg_loss
    
    def analyze_logit_consensus(self, round_num, consensus_logits):
        """로짓 합의 분석"""
        print(f"\n=== Round {round_num} - Logit Consensus Analysis ===")
        
        # 로짓 통계
        logits_array = consensus_logits
        print(f"Consensus logits shape: {logits_array.shape}")
        print(f"Logit mean: {np.mean(logits_array):.4f}")
        print(f"Logit std: {np.std(logits_array):.4f}")
        print(f"Logit min: {np.min(logits_array):.4f}")
        print(f"Logit max: {np.max(logits_array):.4f}")
        
        # 소프트맥스 확률 분포
        softmax_probs = torch.softmax(torch.tensor(logits_array), dim=-1)
        entropy = -torch.sum(softmax_probs * torch.log(softmax_probs + 1e-8), dim=-1)
        avg_entropy = torch.mean(entropy).item()
        print(f"Average entropy (uncertainty): {avg_entropy:.4f}")
        
        # 가장 확신하는 클래스들
        max_probs, predicted_classes = torch.max(softmax_probs, dim=-1)
        avg_confidence = torch.mean(max_probs).item()
        print(f"Average confidence: {avg_confidence:.4f}")
        
        # 클래스 분포
        class_counts = torch.bincount(predicted_classes)
        print(f"Predicted class distribution: {class_counts.tolist()}")
        
        # 클라이언트 간 로짓 차이 분석
        self._analyze_client_logit_differences(round_num, logits_array)
    
    def _analyze_client_logit_differences(self, round_num, consensus_logits):
        """클라이언트 간 로짓 차이 분석"""
        print(f"\n--- Client Logit Differences Analysis ---")
        
        # 각 클라이언트의 로짓 수집
        client_logits = []
        for i, client_proxy in enumerate(self.clients):
            # 동일한 샘플에 대한 로짓 생성
            sample_ids = list(range(min(100, consensus_logits.shape[0])))  # 처음 100개 샘플
            client_logit = client_proxy.client.get_public_logits("cifar10_v1", sample_ids)
            from flwr.common.tensor import tensor_to_ndarray
            logit_array = tensor_to_ndarray(client_logit.logits)
            client_logits.append(logit_array)
            print(f"Client {i+1} logits shape: {logit_array.shape}")
        
        # 클라이언트 간 로짓 차이 계산
        if len(client_logits) > 1:
            consensus_subset = consensus_logits[:len(client_logits[0])]
            differences = []
            for i, client_logit in enumerate(client_logits):
                diff = np.mean(np.abs(client_logit - consensus_subset))
                differences.append(diff)
                print(f"Client {i+1} vs Consensus L1 distance: {diff:.4f}")
            
            avg_difference = np.mean(differences)
            print(f"Average client-consensus difference: {avg_difference:.4f}")
            
            # 클라이언트 간 상호 차이
            client_differences = []
            for i in range(len(client_logits)):
                for j in range(i+1, len(client_logits)):
                    diff = np.mean(np.abs(client_logits[i] - client_logits[j]))
                    client_differences.append(diff)
                    print(f"Client {i+1} vs Client {j+1} L1 distance: {diff:.4f}")
            
            if client_differences:
                avg_client_diff = np.mean(client_differences)
                print(f"Average inter-client difference: {avg_client_diff:.4f}")
                
                # FedMD 효과 분석
                if avg_difference < avg_client_diff:
                    print("✅ FedMD is working: Clients are converging toward consensus!")
                else:
                    print("⚠️  FedMD may not be effective: High client-consensus differences")
    
    def print_summary(self):
        """전체 결과 요약"""
        print("\n" + "="*60)
        print("FEDMD TRAINING SUMMARY")
        print("="*60)
        
        print("\nRound-wise Accuracy:")
        for round_num in sorted(self.round_accuracies.keys()):
            accuracies = self.round_accuracies[round_num]
            avg_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            print(f"Round {round_num}: {avg_acc:.4f} ± {std_acc:.4f}")
        
        print("\nRound-wise Loss:")
        for round_num in sorted(self.round_losses.keys()):
            losses = self.round_losses[round_num]
            avg_loss = np.mean(losses)
            std_loss = np.std(losses)
            print(f"Round {round_num}: {avg_loss:.4f} ± {std_loss:.4f}")
        
        # 개선도 계산
        if len(self.round_accuracies) > 1:
            first_round_acc = np.mean(self.round_accuracies[1])
            last_round_acc = np.mean(self.round_accuracies[max(self.round_accuracies.keys())])
            improvement = last_round_acc - first_round_acc
            print(f"\nAccuracy improvement: {improvement:.4f}")
            
            if improvement > 0:
                print("✅ FedMD training shows positive improvement!")
            else:
                print("⚠️  FedMD training shows no improvement or degradation")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    strategy = build_server_and_strategy()

    # 간단 ClientManager 대체: 리스트로 직접 관리 (app_fedmd 예제와 호환)
    num_clients = 3
    clients = [make_client(device=device) for _ in range(num_clients)]
    
    # FedMD 검증기 초기화
    validator = FedMDValidator(clients)

    # ClientManager mock: 필요한 속성/메서드만 제공
    class _CM:
        def __init__(self, clients):
            self._clients = clients
        def num_available(self): return len(self._clients)
        def sample(self, num_clients): return self._clients[:num_clients]

    cm = _CM(clients)

    print("\n" + "="*60)
    print("STARTING FEDMD TRAINING")
    print("="*60)
    
    # 초기 모델 성능 평가
    print("\n=== Initial Model Performance ===")
    validator.evaluate_client_models(0)

    # FedMD 라운드 3회 실행
    run_fedmd_training(None, strategy, num_rounds=3, client_manager=cm, clients=clients, validator=validator)
    
    # 최종 모델 성능 평가
    print("\n=== Final Model Performance ===")
    validator.evaluate_client_models(4)
    
    # 전체 결과 요약
    validator.print_summary()
    
    print("\nFedMD simulation finished.")

if __name__ == "__main__":
    main()
