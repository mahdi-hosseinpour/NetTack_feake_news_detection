# main.py
import numpy as np
import torch
from src.config import Config
from src.preprocessing import preprocess_liar
from src.features import extract_tfidf_features, reduce_dimensions
from src.graph import build_graph
from src.model import GraphSAGE
from src.train import prepare_data, get_class_weights, train_model, evaluate_model


def set_seed(seed=Config.SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # Set seed for reproducibility
    set_seed(Config.SEED)

    print("=== Starting data preprocessing ===")
    df = preprocess_liar(
        train_path=Config.DATA_DIR / "train.tsv",
        test_path=Config.DATA_DIR / "test.tsv",
        valid_path=Config.DATA_DIR / "valid.tsv"
    )

    print("=== Extracting TF-IDF features ===")
    tfidf_features, tfidf_feature_names, tfidf_vectorizer = extract_tfidf_features(df)

    print("=== Dimensionality reduction with PCA ===")
    reduced_features, pca = reduce_dimensions(tfidf_features)

    print("=== Building graph ===")
    G = build_graph(df, Features_vect, tfidf_feature_names)
    print("=== Preparing data for GNN ===")
    data = prepare_data(G, reduced_features, df['label'].values)

    print("=== Computing class weights ===")
    class_weights = get_class_weights(df['label'].values)

    print("=== Initializing model ===")
    model = GraphSAGE(
        in_channels=reduced_features.shape[1],
        hidden_channels=Config.HIDDEN_DIM,
        out_channels=Config.NUM_CLASSES
    )

    print("=== Training model ===")
    trained_model = train_model(model, data, class_weights)

    print("=== Saving model ===")
    torch.save(trained_model.state_dict(), Config.MODEL_DIR / "graphsage.pth")

    print("=== Evaluating model ===")
    acc, f1_macro, f1_weighted = evaluate_model(trained_model, data)

    print("=== Done ===")
    print(f"Final results: Accuracy: {acc:.4f}, F1-Macro: {f1_macro:.4f}, F1-Weighted: {f1_weighted:.4f}")
    print("\n=== Evaluating baseline model ===")
    base_acc, base_f1_macro, base_f1_weighted = evaluate_model(trained_model, data)

    print("\n=== Nettack Attack ===")
    attacked_acc, attacked_f1_macro, attacked_f1_weighted = evaluate_with_nettack(trained_model, data)

    print(f"Accuracy after Nettack attack: {attacked_acc:.4f} (drop: {base_acc - attacked_acc:.4f})")

    print("\n=== Adversarial Training for robustness ===")
    robust_model = adversarial_training(trained_model, data, num_epochs=100)

    print("\n=== Evaluating robust model under Nettack attack ===")
    robust_attacked_acc, _, _ = evaluate_with_nettack(robust_model, data)

    print(f"Accuracy of robust model after attack: {robust_attacked_acc:.4f}")
    print(f"Robustness improvement: {robust_attacked_acc - attacked_acc:.4f}")

if __name__ == "__main__":
    main()