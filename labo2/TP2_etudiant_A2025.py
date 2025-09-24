# %% [markdown]
# <a href="https://colab.research.google.com/github/eddydq/ELE2700/blob/main/TP2_etudiant_A2025.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# DÃ©partement de gÃ©nie Ã©lectrique - Polytechnique MontrÃ©al
#
# Analyse des signaux - ELE2700
#
# TP2 version 4.0: A2025
#
# ---
#
# NumÃ©ro d'Ã©quipe: Ã€ COMPLÃ‰TER
#
# Noms, prÃ©noms et matricules: Ã€ COMPLÃ‰TER

# %% [markdown]
# # $\color{#18a2f2}{\textbf{TP2 - SÃ©rie de Fourier et Ã©lectronique de puissance}}$

# %%
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift

# %% [markdown]
# ## <font color='lightblue'> Contexte 1: Ã‰tude du redresseur Ã  double alternance </font>

# %% [markdown]
# Les redresseurs Ã  double alternance sont des Ã©quipements Ã©lectroniques utilisÃ©s afin de convertir les parties nÃ©gatives d'une source de tension en entrÃ©e en parties positives. Leur application concerne notamment dans la conception de convertisseurs AC-DC de tension ou de courant.

# %% [markdown]
# ### $\color{#03fc9d}{\textbf{Exercice 1, Ã©valuÃ© en classe (8pt/20):}}$

# %% [markdown]
# Soit l'expression de la tension sinusoÃ¯dale au bornes de la source d'alimentation en entrÃ©e :
# $v_\text{e} \left( t \right) = E \sin \left( 2\pi\frac{t}{T} \right)$.
#
# Soit l'expression mathÃ©matique de la tension en sortie : $v_\text{s} \left( t \right) =  | v_\text{e}(t)|$.

# %% [markdown]
# ### <font color='lightpink'> Partie 1: Formes temporelles des signaux $v_\text{e}(t)$ et $v_\text{s}(t)$ </font>

# %% [markdown]
# $\color{orange}{\text{ Question 1a) [code]}}$

# %% [markdown]
# En considÃ©rant **100 pÃ©riodes** de signaux pour le recueil des donnÃ©es, une fenÃªtre d'observation de **$\left[0,2T\right]$** avec **$ T= \frac{1}{60} \text{s} $** et **une pÃ©riode d'Ã©chantillonage** de **$T_\text{e} = 1~\mu \text{s}$**, tracer **les rÃ©ponses temporelles** des signaux **$\color{red}{v_\text{e}(t)}$** et **$\color{green}{v_\text{s}(t)}$** $\color{red}{\textbf{(2pt/20)}}$

# %%
# Question 1a
T = 1 / 60
Te = 1e-6
nb_periods = 100
E = 120

t = np.arange(0, nb_periods * T, Te)
ve = E * np.sin(2 * np.pi * t / T)
vs = np.abs(ve)

mask = t <= 2 * T
plt.figure(figsize=(9, 3))
plt.plot(t[mask], ve[mask], color="red", label=r"$v_e(t)$")
plt.plot(t[mask], vs[mask], color="green", label=r"$v_s(t)$")
plt.xlabel("Temps (s)")
plt.ylabel("Tension (V)")
plt.legend()
plt.grid(True, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### <font color='lightpink'> Partie 2: SÃ©rie de Fourier du signal $v_\text{s}(t)$ et calcul de la THD </font>

# %% [markdown]
# $\color{orange}{\text{ Question 2a) [Code]}}$

# %% [markdown]
#  Il est possible de dÃ©montrer que les coefficients du dÃ©veloppement en sÃ©rie de Fourier du signal $v_\text{s}(t)$ vÃ©rifient les relations suivantes :
#
# \begin{align*}
# A_0 &= \frac{2E}{\pi}\\
# A_{n\neq 1} &= \frac{2E}{\pi} \frac{1+(-1)^n}{1-n^2}\\
# A_{1} &= 0\\
# B_{n} &= 0 \\
# \end{align*}
#
# Reconstruire le signal **$\color{green}{v_\text{s}(t)}$** Ã  partir d'au moins ses $40$ premiÃ¨res harmoniques **non nulles** de sa dÃ©composition en sÃ©rie de Fourier et le comparer au signal de la question prÃ©cÃ©dente. $\color{red}{\textbf{(2pt/20)}}$

# %%
# Question 2a
T = 1 / 60
E = 1

harmonics = np.arange(2, 2 * 40 + 2, 2)
A0 = 2 * E / np.pi
An = (4 * E / np.pi) / (1 - harmonics**2)

vs_fourier = np.full_like(t, A0 / 2)
for n, an in zip(harmonics, An):
    vs_fourier += an * np.cos(2 * np.pi * n * t / T)

mask = t <= 2 * T
plt.figure(figsize=(9, 3))
plt.plot(t[mask], vs[mask], color="green", label=r"$v_s(t)$")
plt.plot(
    t[mask], vs_fourier[mask], color="blue", linestyle="--", label="Serie de Fourier"
)
plt.xlabel("Temps (s)")
plt.ylabel("Tension (V)")
plt.legend()
plt.grid(True, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()

print(f"Harmoniques non nulles utilisees : {len(harmonics)}")

# %% [markdown]
# $\color{orange}{\text{ Question 2b) [Code]}}$

# %% [markdown]
# Superposer un tracÃ© thÃ©orique et celui obtenu grÃ¢ce aux outils numÃ©riques du spectre du signal **$\color{green}{v_\text{s}(t)}$** en allant jusqu'Ã  l'harmonique de dÃ©grÃ© $12$. $\color{red}{\textbf{(2pt/20)}}$

# %% [markdown]
# > **Quelques recommandations** :
# > * Pour le tracÃ© thÃ©orique, pour chacune des frÃ©quences du spectre, la magnitude de la frÃ©quence caractÃ©risÃ©e par $n$ se calcule tel que suit : $\sqrt{A_n^2+B_n^2}$ oÃ¹ $n$ reprÃ©sente le multiple de la frÃ©quence fondamentale
# > * Vous devrez prÃ©ciser l'attribut `norm='forward'` afin de normaliser les spectres de Fourier obtenus via `scipy` dans la fonction `scipy.fft`;
# > * Pour assurer une cohÃ©rence entre les magntitudes de la sÃ©rie de Fourier obtenues analytiquement et celles obtenues via l'outil numÃ©rique, vous devrez multiplier vos normes analytiques par un facteur $\frac{1}{2}$ (hormis celle de la composante de frÃ©quence nulle du signal, ceci dÃ©coule des diffÃ©rentes dÃ©finition du DSF);
# > * Le tracÃ© dans les axes des frÃ©quences nÃ©gatives se fait par une rÃ©flexion par rapport Ã  l'axe des ordonnÃ©es du spectre tracÃ© dans les frÃ©quences positives.

# %%
# Question 2b
T = 1 / 60
E = 120
Te = t[1] - t[0]
f0 = 1 / T

Vs = fft(vs, norm="forward")
freqs = np.fft.fftfreq(t.size, Te)

indices = []
n_values = np.arange(0, 13)
for n in n_values:
    freq_target = n * f0
    indices.append(np.argmin(np.abs(freqs - freq_target)))

magn_fft = np.abs(Vs[indices])


def fourier_A(n):
    if n == 0:
        return 2 * E / np.pi
    if n == 1:
        return 0.0
    return (2 * E / np.pi) * (1 + (-1) ** n) / (1 - n**2)


magn_theo = []
for n in n_values:
    value = fourier_A(n)
    if n == 0:
        magn_theo.append(np.abs(value))
    else:
        magn_theo.append(np.abs(value) / 2)

plt.figure(figsize=(9, 3))
plt.stem(
    n_values * f0,
    magn_fft,
    linefmt="C2-",
    markerfmt="C2o",
    basefmt="k-",
    label="FFT (norm forward)",
)
plt.stem(
    n_values * f0,
    magn_theo,
    linefmt="C1--",
    markerfmt="C1s",
    basefmt="k-",
    label="Serie analytique",
)
plt.xlabel("Frequence (Hz)")
plt.ylabel("Magnitude")
plt.title("Spectre de v_s(t) jusqu'a l'harmonique 12")
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# $\color{orange}{\text{ Question 2c) [DÃ©marche manuscrite ou } \LaTeX \text{]}}$

# %% [markdown]
# En se basant sur vos rÃ©sultats, quelle est la frÃ©quence fondamentale du signal **$\color{green}{v_\text{s}(t)}$**? Comment compare-t-elle Ã  la frÃ©quence du signal **$\color{red}{v_\text{e}(t)}$**? Reformuler les coefficients de la sÃ©rie de Fourier donnÃ©s Ã  la question 2a) en utilisant la plus petite pÃ©riode possible. $\color{red}{\textbf{(2pt/20)}}$

# %% [markdown]
# ---
# ---
# InsÃ©rez votre dÃ©marche ici.
#
# ---
# ---

# %% [markdown]
# ### $\color{#03fc9d}{\textbf{Exercice 2, Ã©valuÃ© en devoir (12pt/20):}}$

# %% [markdown]
# ## <font color='lightblue'> Contexte 2: Ã‰tude du convertisseur AC-DC </font>

# %% [markdown]
# Le redresseur Ã  double alternance est une composante importante pour la conception d'un convertisseur AC-DC. Cependant, le signal Ã  sa sortie comprends encore des fortes oscillations, le systÃ¨me est alors incomplet.

# %% [markdown]
# ### <font color='lightpink'> Partie 3: SÃ©rie de Fourier du signal $v_s$ </font>

# %% [markdown]
# $\color{orange}{\text{ Question 3a) [DÃ©marche manuscrite ou } \LaTeX \text{]}}$ DÃ©montrer comment obtenir les coefficients du dÃ©veloppement en sÃ©rie de Fourier prÃ©sentÃ©s Ã  la question Q2a pour le redresseur Ã  double alternance. $\color{red}{\textbf{(4pt/20)}}$

# %% [markdown]
# ---
# ---
# InsÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â©rez votre dÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â©marche ici.
#
# ---
# ---

# %% [markdown]
# ### <font color='lightpink'> Partie 4: Conception du filtre </font>

# %% [markdown]
# Il est possible d'obtenir une tension continue Ã  partir du signal $\color{green}{v_\text{s}(t)}$ en utilisant un filtre. Le filtre suivant sera utilisÃ© Ã  cet effet:
#
# ![RC_Circuit.png](data:image/png;base64,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)

# %% [markdown]
#
# La rÃ©ponse en frÃ©quence ce filtre peut Ãªtre Ã©crite de la faÃ§on suivante:
#
# \begin{equation*}
#     H(f) = \frac{1}{1 + j 2 \pi f R C}
# \end{equation*}
#
# $\color{orange}{\text{ Question 4a) [RÃ©ponse Ã©crite]}}$ Quel est le type de ce filtre? En observant le spectre du signal $\color{green}{v_\text{s}(t)}$, quelle frÃ©quence(s) faut-il conserver pour obtenir une tension constante? $\color{red}{\textbf{(1pt/20)}}$

# %% [markdown]
# # Ã‰crire votre code ici pour (Q4d)

# %% [markdown]
# $\color{orange}{\text{ Question 4b) [DÃ©marche manuscrite ou } \LaTeX \text{]}}$ Reformuler les coefficients du dÃ©veloppement en sÃ©rie de Fourier prÃ©sentÃ©s Ã  la question Q2a) pour inclure l'impact de ce filtre. Utiliser la forme exponentielle et simplifiez votre rÃ©ponse. $\color{red}{\textbf{(2pt/20)}}$

# %% [markdown]
# ---
# ---
# InsÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â©rez votre dÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â©marche ici.
#
# ---
# ---

# %% [markdown]
# $\color{orange}{\text{ Question 4c) [DÃ©marche manuscrite ou } \LaTeX \text{]}}$ Vous dÃ©cidez de reproduire ce circuit en laboratoire pour valider vos calculs. Vous dÃ©cidez d'utiliser un condensateur de 47 $\mu\mathrm{F}$. Quelle est la valeur de la resistance nÃ©cessaire pour que l'amplitude de la premiÃ¨re harmonique non-nulle soit 50 fois plus petite que la composante constante? Utiliser la base exponentielle. $\color{red}{\textbf{(2pt/20)}}$

# %% [markdown]
# ---
# ---
# InsÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â©rez votre dÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â©marche ici.
#
# ---
# ---

# %% [markdown]
# $\color{orange}{\text{ Question 4d) [code]}}$
#
# Reconstruire le signal **$\color{blue}{v_\text{c}(t)}$** Ã  partir d'au moins ses $40$ premiÃ¨res harmoniques **non nulles** de sa dÃ©composition en sÃ©rie de Fourier. Afficher sur la mÃªme figure le signal $\color{green}{v_\text{s}(t)}$ et comparer. Afficher un tracÃ© thÃ©orique du spectre en allant jusqu'Ã  l'harmonique de degrÃ© 12.  $\color{red}{\textbf{(2pt/20)}}$

# %%
# ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°crire votre code ici pour (Q4d)


# %% [markdown]
# $\color{orange}{\text{ Question 4e) [RÃ©ponse Ã©crite]}}$ Commentez sur la forme du signal **$\color{blue}{v_\text{c}(t)}$** et sur son spectre. Comment pouvez-vous amÃ©liorer la qualitÃ© du signal? $\color{red}{\textbf{(1pt/20)}}$

# %% [markdown]
# #Reponse 4e)

# %% [markdown]
# ## $\color{#03fc9d}{\textbf{PrÃ©cisions pour la remise:}}$
#
# * Pour chaque exercice, rÃ©pondre en ajoutant des cases de code ou de texte en dessous de la question.
#
# * Remettre sur moodle un fichier `.ipynb` contenant tous les exercises de ce TP avec toutes les librairies et importations nÃ©cessaires pour que le code roule sans erreurs. Remettre Ã©galement un fichier `.pdf` du notebook. (Voir Notes additionnelles du TP-1)
#
# * Indiquer vos noms, vos matricules et votre numÃ©ro d'Ã©quipe au dÃ©but dans l'entÃªte du `notebook`.
