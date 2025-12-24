# Salvar métricas

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def compute_metrics_and_confmat(y_true, y_probs, threshold=0.5):
    y_pred = (y_probs >= threshold).astype(int)
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = int(cm[0,0]), int(cm[0,1]), int(cm[1,0]), int(cm[1,1])
    tnr = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    return {
        'accuracy': acc,
        'precision': prec,
        'true_negative_rate': float(tnr),
        'recall': rec,
        'f1_score': f1
    }, {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}

print('Escolha de métricas pronta')
