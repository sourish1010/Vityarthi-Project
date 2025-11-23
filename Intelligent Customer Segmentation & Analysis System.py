# mall_customers_like_a_real_person.py
# This is the one I actually run when I want to feel good about code

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pickle
import os

# soft colours, nothing too serious
plt.style.use('seaborn-v0_8')
sns.set_palette("pastel")


class ShopAssistant:
    def __init__(self):
        self.datafile = "Mall_Customers.csv"
        self.df = None
        self.model = None
        self.scaler = StandardScaler()
        self.ready = None  # will hold scaled features

        # names I actually like saying out loud
        self.nicknames = {
            0: "Careful Earners – make good money, spend wisely",
            1: "Young & Fun – not rich yet but live a little",
            2: "Big Spenders – high income + high taste = best customers",
            3: "Sensible Savers – careful with every dollar",
            4: "Comfortable Middle – steady income, steady spending, happy"
        }

    def hi(self):
        print("\nHey there,")
        print("I’m the person who turns a spreadsheet of mall customers into something that actually makes sense.")
        print("Ready when you are.\n")

    def open_the_file(self):
        if not os.path.exists(self.datafile):
            print(f"Uh oh – I can't see {self.datafile}")
            print("Pop it in the same folder as me and try again, yeah?")
            return False

        self.df = pd.read_csv(self.datafile)
        print(f"Found it! {len(self.df)} customers waiting to be understood.")
        print(f"Ages {self.df.Age.min()} to {self.df.Age.max()} – proper mix.\n")
        return True

    def quick_chat_about_them(self):
        print("Quick look at who we've got:\n")
        ladies = (self.df.Gender == "Female").sum()
        gents = (self.df.Gender == "Male").sum()
        print(f"→ {ladies} women, {gents} men (ladies win, as usual)")
        print(f"→ average age around {self.df.Age.mean():.0f}")
        print(f"→ income from ${self.df['Annual Income (k$)'].min()}k to ${self.df['Annual Income (k$)'].max()}k")
        print(f"→ average spending score {self.df['Spending Score (1-100)'].mean():.1f}/100\n")

    def get_ready(self):
        print("Just making sure the numbers play fair with each other...")
        features = self.df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
        self.ready = self.scaler.fit_transform(features)
        print("All good.\n")

    def elbow_time(self):
        print("Trying a few group sizes to see what feels right...")
        scores = []
        for k in range(2, 9):
            temp = KMeans(n_clusters=k, random_state=42, n_init=10)
            temp.fit(self.ready)
            scores.append(temp.inertia_)

        plt.figure(figsize=(9, 5))
        plt.plot(range(2, 9), scores, 'o-', color='#3498db', linewidth=2, markersize=8)
        plt.title("How many groups makes sense?", fontsize=14, pad=20)
        plt.xlabel("Number of groups")
        plt.ylabel("How tight the groups are")
        plt.axvline(5, color='coral', linestyle='--', alpha=0.8)
        plt.text(5.1, max(scores) * 0.8, "5 usually feels spot-on", color='coral', fontsize=11)
        plt.grid(alpha=0.3)
        plt.show()

        print("I nearly always end up with 5 – it just works.")
        answer = input("Stick with 5? (press Enter) or type another number → ").strip()
        return int(answer) if answer.isdigit() else 5

    def do_the_magic(self, k=5):
        print(f"\nOkay, let’s sort everyone into {k} types of shopper...")
        self.model = KMeans(n_clusters=k, random_state=42, n_init=10)
        self.df['group'] = self.model.fit_predict(self.ready)
        self.df['type'] = self.df['group'].map(self.nicknames)
        print("Done. Every customer now has a little label that actually means something.\n")

    def introduce_the_families(self):
        print("=" * 65)
        print("HERE ARE YOUR CUSTOMER FAMILIES")
        print("=" * 65)

        summary = self.df.groupby('type').agg({
            'Age': 'mean',
            'Annual Income (k$)': 'mean',
            'Spending Score (1-100)': 'mean',
            'CustomerID': 'count'
        }).round(1)

        summary['pct'] = (summary.CustomerID / len(self.df) * 100).round(1)
        summary = summary.sort_values('CustomerID', ascending=False)

        for person, row in summary.iterrows():
            short_name = person.split(" – ")[0]
            description = person.split(" – ")[1]
            print(f"\n{short_name}")
            print(f"   → {int(row.CustomerID)} customers ({row.pct}%)")
            print(f"   → avg age {row.Age:.0f} | income ${row['Annual Income (k$)']:.0f}k | spending {row['Spending Score (1-100)']:.0f}/100")
            print(f"   → {description}")

    def pretty_pictures(self):
        print("\nDrawing everything so you can actually see it...")

        fig = plt.figure(figsize=(18, 11))
        fig.suptitle("Your customers – now in living colour", fontsize=20, fontweight='bold', y=0.95)

        # income vs spending – the classic
        plt.subplot(2, 3, 1)
        plt.scatter(self.df['Annual Income (k$)'], self.df['Spending Score (1-100)'],
                   c=self.df.group, cmap='Set2', s=90, edgecolor='white', linewidth=0.5, alpha=0.9)
        plt.title("Income vs How Much They Love Spending", fontweight='bold')
        plt.xlabel("Annual Income (k$)")
        plt.ylabel("Spending Score")

        # age vs spending
        plt.subplot(2, 3, 2)
        plt.scatter(self.df.Age, self.df['Spending Score (1-100)'],
                   c=self.df.group, cmap='Set2', s=90, edgecolor='white', linewidth=0.5, alpha=0.9)
        plt.title("Younger ones definitely spend more freely", fontweight='bold')
        plt.xlabel("Age")

        # pie chart – because managers love pie charts
        plt.subplot(2, 3, 3)
        sizes = self.df['type'].value_counts()
        labels = [name.split(" – ")[0] for name in sizes.index]
        plt.pie(sizes, labels=labels, colors=sns.color_palette("pastel"), autopct='%1.0f%%', startangle=90)
        plt.title("How big each family is")

        # 3D view – because it looks cool
        ax = fig.add_subplot(2, 3, (4, 6), projection='3d')
        scatter = ax.scatter(self.df['Annual Income (k$)'],
                            self.df['Spending Score (1-100)'],
                            self.df.Age,
                            c=self.df.group, cmap='Set2', s=70, alpha=0.8)
        ax.set_xlabel('Income', labelpad=10)
        ax.set_ylabel('Spending', labelpad=10)
        ax.set_zlabel('Age', labelpad=10)
        ax.set_title("Same people, fancier angle", fontweight='bold', pad=20)

        plt.tight_layout()
        plt.show()

    def who_is_this_person(self):
        print("\nGot someone new walking in?")
        try:
            print("Tell me a bit about them:")
            age = int(input("   Age → "))
            income = int(input("   Rough income (thousands) → "))
            score = int(input("   Spending score 1–100 → "))

            person = self.scaler.transform([[age, income, score]])
            group = self.model.predict(person)[0]
            name = self.nicknames[group].split(" – ")[0]

            print(f"\nAh yes – they’re definitely one of the {name}")
            print("Look after them")
        except:
            print("\nNo worries, maybe next time")

    def remember_for_next_time(self):
        with open("mall_memory.pkl", "wb") as f:
            pickle.dump((self.model, self.scaler, self.nicknames), f)
        print("\nI’ve saved everything I learned. I won’t forget your people.")

    def run(self):
        self.hi()
        if not self.open_the_file():
            return

        self.quick_chat_about_them()
        self.get_ready()
        k = self.elbow_time()
        self.do_the_magic(k)
        self.introduce_the_families()
        self.pretty_pictures()
        self.who_is_this_person()
        self.remember_for_next_time()

        print("\nThat’s it.")
        print("You don’t have numbers anymore – you have people.")
        print("Go be kind to them")


# ————————————————————
if __name__ == "__main__":
    assistant = ShopAssistant()
    assistant.run()