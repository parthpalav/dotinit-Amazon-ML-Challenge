"""Generate fictional synthetic fixtures; never represents competition data."""
import argparse
from pathlib import Path
import random
import string
import pandas as pd

COLUMNS = ["entity_id", "business_name", "business_address", "country"]


def make_demo(root: Path, count: int = 100):
    randomizer = random.Random(817)
    for split in ("train", "test"):
        directory = root / split
        directory.mkdir(parents=True, exist_ok=True)
        if any(directory.glob("*.tsv")):
            raise FileExistsError(f"Refusing to overwrite existing TSVs in {directory}")
        anchors, source2, source3, truth = [], [], [], []
        for i in range(count):
            token = "".join(randomizer.choices(string.ascii_lowercase, k=10))
            source = f"S1-{split}-{i:05d}"
            # Countries are fabricated, with one unseen test value and missing values.
            country = ("Novel Republic" if split == "test" and i % 7 == 0 else ["Northland", "Coast Union", "南国", ""][i % 4])
            address = f"{10+i} {token} Road, City {i%11}, Region {i%3} {20000+i}"
            name = f"{token} Workshop"
            anchors.append([source, name + " Ltd.", address, country])
            actual = []
            if i % 4 != 0:
                target = f"S2-{split}-match-{i:05d}"
                source2.append([target, name + " Limited", address.replace("Road", "Rd."), country])
                actual.append(target)
                if i % 3 == 0:
                    target = f"S3-{split}-match-{i:05d}"
                    source3.append([target, name.replace(token, token[:-1]), address, country])
                    actual.append(target)
            # Hard name negatives; singleton anchors also receive plausible candidates.
            source2.append([f"S2-{split}-hard-{i:05d}", token + " Workshop Supplies",
                            f"{700+i} Elsewhere Ave, Other City, Far Region {80000+i}", country])
            # Hard address negative at the same premises and postal code.
            source3.append([f"S3-{split}-address-{i:05d}", f"Unrelated Tenant {i}", address, country])
            source3.append([f"S3-{split}-easy-{i:05d}", f"Distinct Firm {i}", f"{9000+i} Remote Lane", country])
            truth.append([source, ",".join(actual)])
        for source, rows in enumerate((anchors, source2, source3), 1):
            pd.DataFrame(rows, columns=COLUMNS).to_csv(directory / f"{split}_source{source}.tsv", sep="\t", index=False)
        if split == "train":
            pd.DataFrame(truth, columns=["source1_entity_id", "matched_entity_ids"]).to_csv(
                directory / "train_ground_truth.tsv", sep="\t", index=False)
    (root / "SYNTHETIC_DATA.txt").write_text("Fictional generated data for software testing only. Scores do not estimate challenge performance.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("demo/dataset"))
    parser.add_argument("--count", type=int, default=100)
    arguments = parser.parse_args()
    if arguments.count < 10:
        parser.error("--count must be at least 10")
    make_demo(arguments.root, arguments.count)
    print(f"Synthetic fixtures created in {arguments.root}")
