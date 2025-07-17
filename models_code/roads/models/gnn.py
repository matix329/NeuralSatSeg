import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models_code.roads.metrics.metrics_graph import edge_precision, edge_recall, edge_f1, edge_iou, edge_mae, edge_rmse, edge_spearman

class RoadGNN(nn.Module):
    def __init__(self, in_channels=2, hidden_channels=32, heads=2, dropout=0.2):
        super().__init__()
        self.gcn1 = GCNConv(in_channels, hidden_channels)
        self.gat = GATConv(hidden_channels, hidden_channels, heads=heads, concat=False, dropout=dropout)
        self.gcn2 = GCNConv(hidden_channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gcn1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.gat(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.gcn2(x, edge_index)
        x = x.view(-1)
        return x

def create_gnn(*args, **kwargs):
    return RoadGNN(*args, **kwargs)

def train_gnn(model, train_loader, val_loader, config, logger, mlflow_callback, epoch_logger):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    learning_rate = config["learning_rate"]
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    total_positives = 0
    total_negatives = 0
    for _, label in train_loader:
        label = label.view(-1)
        total_positives += (label == 1).sum().item()
        total_negatives += (label == 0).sum().item()
    total = total_positives + total_negatives
    pos_ratio = total_positives / total if total > 0 else 0.0
    logger.info(f"Class distribution: positives={total_positives}, negatives={total_negatives}, pos_ratio={pos_ratio:.4f}")
    if pos_ratio < 0.05:
        logger.warning("Highly imbalanced classes: <5% positive class! Check your masks or consider oversampling/undersampling.")
    if total_positives == 0:
        pos_weight = torch.tensor(1.0, device=device)
    else:
        pos_weight = torch.tensor(total_negatives / total_positives, device=device)
    logger.info(f"pos_weight for BCEWithLogitsLoss: {pos_weight.item():.4f}")
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    best_val_loss = float('inf')
    patience_counter = 0
    if "epochs" not in config:
        raise ValueError("Missing 'epochs' parameter in config! Add it to config_graph.json.")
    if "patience" not in config:
        raise ValueError("Missing 'patience' parameter in config! Add it to config_graph.json.")
    EPOCHS = config["epochs"]
    PATIENCE = config["patience"]
    f1_history = []
    iou_history = []
    val_loss_history = []
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for batch_idx, (data, label) in enumerate(train_loader):
            data = data.to(device)
            label = label.to(device)
            label = label.view(-1)
            if epoch == 0 and batch_idx == 0:
                logger.info(f"data.x shape: {data.x.shape}, data.edge_index shape: {data.edge_index.shape}")
                if data.x.shape[0] == 0 or data.edge_index.shape[1] == 0:
                    logger.warning("Empty graph detected! Check your .pt files and mask generation.")
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_loss = epoch_loss / len(train_loader)
        model.eval()
        val_loss = 0
        preds = []
        gts = []
        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(val_loader):
                data = data.to(device)
                label = label.to(device)
                label = label.view(-1)
                out = model(data)
                loss = criterion(out, label)
                val_loss += loss.item()
                probs = torch.sigmoid(out.detach().cpu())
                preds.append(probs)
                gts.append(label.detach().cpu())
        val_loss /= len(val_loader)
        preds = torch.cat(preds)
        gts = torch.cat(gts)
        prec = edge_precision(preds, gts)
        rec = edge_recall(preds, gts)
        f1 = edge_f1(preds, gts)
        iou = edge_iou(preds, gts)
        mae = edge_mae(preds, gts)
        rmse = edge_rmse(preds, gts)
        spearman = edge_spearman(preds, gts)
        logs = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "edge_precision": prec,
            "edge_recall": rec,
            "edge_f1": f1,
            "edge_iou": iou,
            "edge_mae": mae,
            "edge_rmse": rmse,
            "edge_spearman": spearman
        }
        epoch_logger.on_epoch_end(epoch, logs)
        mlflow_callback.on_epoch_end(epoch, logs)
        logger.info(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, edge_f1={f1:.4f}, edge_iou={iou:.4f}")
        f1_history.append(f1)
        iou_history.append(iou)
        val_loss_history.append(val_loss)
        if epoch >= 5 and all(f == 0.0 for f in f1_history[-5:]):
            logger.warning("edge_f1 == 0.0 for the last 5 epochs! Check class distribution, masks or graph generation.")
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_road_gnn.pt")
            logger.info("Model checkpoint saved.")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info("Early stopping!")
                break
    logger.info("\nEpoch | edge_f1 | edge_iou | val_loss")
    for i, (f1, iou, vloss) in enumerate(zip(f1_history, iou_history, val_loss_history)):
        logger.info(f"{i+1:5d} | {f1:7.4f} | {iou:8.4f} | {vloss:8.4f}") 