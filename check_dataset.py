from pathlib import Path

train_imgs = list(Path("output/images/train").glob("*.*"))
val_imgs = list(Path("output/images/val").glob("*.*"))
test_imgs = list(Path("output/images/test").glob("*.*"))

train_labels = list(Path("output/labels/train").glob("*.txt"))
val_labels = list(Path("output/labels/val").glob("*.txt"))
test_labels = list(Path("output/labels/test").glob("*.txt"))

print("Train Images:", len(train_imgs))
print("Train Labels:", len(train_labels))

print("Val Images:", len(val_imgs))
print("Val Labels:", len(val_labels))

print("Test Images:", len(test_imgs))
print("Test Labels:", len(test_labels))