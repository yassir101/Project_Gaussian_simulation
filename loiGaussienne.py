#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np
import matplotlib.pyplot as plt

def simulate_gaussian_direct(mu, sigma, size=1):
    # Générer des échantillons selon une distribution normale standard (N(0, 1))
    standard_normal_samples = np.random.normal(0, 1, size)
    
    # Appliquer la transformation pour obtenir une gaussienne de moyenne mu et variance sigma^2
    gaussian_samples = mu + sigma * standard_normal_samples
    
    return gaussian_samples

def simulate_gaussian_box_muller(mu, sigma, size=1):
    # Générer des échantillons selon la méthode de Box-Muller
    u1 = np.random.uniform(0, 1, size)
    u2 = np.random.uniform(0, 1, size)
    z0 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
    
    # Appliquer la transformation pour obtenir une gaussienne de moyenne mu et variance sigma^2
    gaussian_samples = mu + sigma * z0
    
    return gaussian_samples

# Exemple d'utilisation
mu = 2.0
sigma = 1.5
size = 1000

# Générer des échantillons selon la méthode de transformation directe
gaussian_samples_direct = simulate_gaussian_direct(mu, sigma, size)

# Générer des échantillons selon la méthode de Box-Muller
gaussian_samples_box_muller = simulate_gaussian_box_muller(mu, sigma, size)

# Afficher quelques statistiques des échantillons simulés
print("Moyenne empirique (Transformation directe) :", np.mean(gaussian_samples_direct))
print("Variance empirique (Transformation directe) :", np.var(gaussian_samples_direct))
print("\nMoyenne empirique (Box-Muller) :", np.mean(gaussian_samples_box_muller))
print("Variance empirique (Box-Muller) :", np.var(gaussian_samples_box_muller))

# Afficher les fonctions de répartition empiriques
sorted_samples_direct = np.sort(gaussian_samples_direct)
cumulative_prob_direct = np.arange(1, size + 1) / size

sorted_samples_box_muller = np.sort(gaussian_samples_box_muller)
cumulative_prob_box_muller = np.arange(1, size + 1) / size

plt.figure(figsize=(12, 6))
plt.plot(sorted_samples_direct, cumulative_prob_direct, label='Fonction de répartition empirique (Transformation directe)')
plt.plot(sorted_samples_box_muller, cumulative_prob_box_muller, label='Fonction de répartition empirique (Box-Muller)')
plt.title('Comparaison des Fonctions de Répartition Empiriques de Gaussiennes')
plt.xlabel('Valeurs de la variable aléatoire')
plt.ylabel('Probabilité cumulée')
plt.text(mu, 0.8, f'$\mu = {mu}$', verticalalignment='bottom', horizontalalignment='right', color='red', fontsize=12)
plt.text(mu, 0.7, f'$\sigma = {sigma}$', verticalalignment='bottom', horizontalalignment='right', color='green', fontsize=12)
plt.legend()
plt.show()


# In[25]:


# Générer des échantillons selon la méthode de transformation directe
gaussian_samples_direct = simulate_gaussian_direct(mu, sigma, size)

# Générer des échantillons selon la méthode de Box-Muller
gaussian_samples_box_muller = simulate_gaussian_box_muller(mu, sigma, size)

# Afficher les densités de probabilité empiriques
x_values = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
pdf_direct = norm.pdf(x_values, mu, sigma)
pdf_box_muller = norm.pdf(x_values, mu, sigma)

plt.figure(figsize=(12, 6))
plt.plot(x_values, pdf_direct, label='Densité de probabilité empirique (Transformation directe)')
plt.plot(x_values, pdf_box_muller, label='Densité de probabilité empirique (Box-Muller)')
plt.title('Comparaison des Densités de Probabilité Empiriques de Gaussiennes')
plt.xlabel('Valeurs de la variable aléatoire')
plt.ylabel('Densité de probabilité')
plt.text(mu, 0.1, f'$\mu = {mu}$, $\sigma = {sigma}$', verticalalignment='bottom', horizontalalignment='right', color='black', fontsize=12)
plt.legend()
plt.show()


# In[26]:


# Générer des échantillons selon la méthode de transformation directe
gaussian_samples_direct = simulate_gaussian_direct(mu, sigma, size)

# Générer des échantillons selon la méthode de Box-Muller
gaussian_samples_box_muller = simulate_gaussian_box_muller(mu, sigma, size)

# Afficher les distributions empiriques
plt.figure(figsize=(12, 6))
plt.hist(gaussian_samples_direct, bins=30, density=True, alpha=0.5, label='Distribution empirique (Transformation directe)')
plt.hist(gaussian_samples_box_muller, bins=30, density=True, alpha=0.5, label='Distribution empirique (Box-Muller)')
plt.title('Comparaison des Distributions Empiriques de Gaussiennes')
plt.xlabel('Valeurs de la variable aléatoire')
plt.ylabel('Fréquence')
plt.text(mu, 0.1, f'$\mu = {mu}$, $\sigma = {sigma}$', verticalalignment='bottom', horizontalalignment='right', color='black', fontsize=10)
plt.legend()
plt.show()


# In[27]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def box_muller(mu, sigma, size=1000):
    # Générer des échantillons selon la méthode de Box-Muller
    u1 = np.random.uniform(0, 1, size)
    u2 = np.random.uniform(0, 1, size)
    z0 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
    
    # Appliquer la transformation pour obtenir une gaussienne de moyenne mu et variance sigma^2
    gaussian_samples = mu + sigma * z0
    
    return gaussian_samples

# Paramètres pour différentes gaussiennes
parameters = [(0, 1), (2, 0.5), (-1, 2)]

# Afficher les densités de probabilité empiriques
plt.figure(figsize=(12, 6))

for mu, sigma in parameters:
    gaussian_samples_box_muller = box_muller(mu, sigma, size=10000)
    x_values = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
    pdf_box_muller = norm.pdf(x_values, mu, sigma)
    
    plt.plot(x_values, pdf_box_muller, label=f'Distribution empirique ($\mu={mu}$, $\sigma={sigma}$)')

plt.title('Densités de Probabilité de Gaussiennes générées par Box-Muller')
plt.xlabel('Valeurs de la variable aléatoire')
plt.ylabel('Densité de probabilité')
plt.legend()
plt.show()


# In[ ]:




