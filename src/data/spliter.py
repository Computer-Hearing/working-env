from sklearn.model_selection import train_test_split

def split_data(data, labels_category, labels_target, test_size=0.15, val_size=0.15, seed=42):
    """
    Разделяет данные на train/val/test
    """
    train_data, test_data, train_category, test_category, train_target, test_target = train_test_split(
        data, labels_category, labels_target, test_size=test_size, random_state=seed
    ) # первый раз разбиваем данные на train и test

    train_data, val_data, train_category, val_category, train_target, val_target = train_test_split(
        train_data, train_category, train_target, test_size=val_size, random_state=seed
    ) # разбиваем train из предыдующего шага на еще один train и еще на val data
    # В итоге получается train, val и test выборки
    train_dataset = (train_data, train_category, train_target)
    val_dataset = (val_data, val_category, val_target)
    test_dataset = (test_data, test_category, test_target)

    return train_dataset, val_dataset, test_dataset
