import torch
import torch.nn as nn

class NLLSurvLoss(nn.Module):
    def __init__(self, epsilon=1e-7):
        super(NLLSurvLoss, self).__init__()
        self.epsilon = epsilon  # Pour éviter log(0) 

    def forward(self, risk_death_predict, target):
        """
        Calcule la Negative Log-Likelihood pour des données de survie discrètes.

        Args:
            risk_death_predict (torch.Tensor): Sortie du modèle (Batch, Max_Time).
                                  Représente P(T=t).
            target (torch.Tensor): labels (Batch, 2).
                                   Col 0 = Temps (Index), Col 1 = Event (0/1).
        """
        # On sépare les temps et les indicateurs d'événement
        times = target[:, 0].long().unsqueeze(1)  # Convert to int64 and add dimension
        # le unsqueeze permet de rajouter une dimension. Nécessaire pour utiliser gather ensuite
        events = target[:, 1] # 1 = Mort, 0 = Censuré

        # Patient décédés : on prend le risques au temps d'évènement
        risk_at_times_to_event = risk_death_predict.gather(1, times).squeeze()


        # Patients censurés : On calcule la survie jusqu'au temps d'évènement
        survival_function = 1.0 - torch.cumsum(risk_death_predict, dim=1)
        survival_at_times_to_event = survival_function.gather(1, times).squeeze()

        # Attention : on va prendre le log, on s'assure alors que c'est pas trop proche de 0
        survival_at_times_to_event = torch.clamp(survival_at_times_to_event, min=self.epsilon)
        risk_at_times_to_event = torch.clamp(risk_at_times_to_event, min=self.epsilon)

        # On applique le log
        uncensored_loss = -torch.log(risk_at_times_to_event)
        censored_loss = -torch.log(survival_at_times_to_event)

        # On applique l'indicateur d'évènement pour choisir la bonne loss par patient
        loss = events * uncensored_loss + (1.0 - events) * censored_loss

        # On renvoie la somme (on peut aussi renvoyer la moyenne si on préfère)
        return loss.sum()