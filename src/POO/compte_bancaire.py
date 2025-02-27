class CompteBancaire:
    titulaire = ""
    solde = 0

    def depose(self, montant):
        self.solde = self.solde + montant

    def retirer(self, montant):
        self.solde = self.solde - montant

    def afficher_solde(self):
        print("Solde actuel du compte bancaire : " + str(self.solde))
