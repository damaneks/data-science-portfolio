from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class RemoveUnitInformation(BaseEstimator, TransformerMixin):
    def __init__(self, eur_to_pln=4.6):
        self.eur_to_pln = eur_to_pln

    def fit(self, X):
        return self

    def transform(self, X):
        X_ = X.copy()

        pln_data = X_.index[X_['Cena'].str.contains('PLN')]
        eur_data = X_.index[X_['Cena'].str.contains('EUR')]
        X_.update(X_['Cena'].loc[pln_data].apply(lambda x: float(
            str(x).rstrip('PLN').replace(' ', '').replace(',', '.'))))
        X_.update(X_['Cena'].loc[eur_data].apply(lambda x: float(
            str(x).rstrip('EUR').replace(' ', '').replace(',', '.')) * self.eur_to_pln))
        X_['Cena'] = X_['Cena'].astype(float)

        X_.update(X_['Przebieg'].apply(lambda x: float(
            str(x).rstrip('km').replace(' ', ''))))
        X_['Przebieg'] = X_['Przebieg'].astype(float)

        X_.update(X_['Pojemność skokowa'].apply(
            lambda x: float(str(x).rstrip('cm3').replace(' ', ''))))
        X_['Pojemność skokowa'] = X_['Pojemność skokowa'].astype(float)

        X_.update(X_['Moc'].apply(lambda x: float(
            str(x).rstrip('KM').replace(' ', ''))))
        X_['Moc'] = X_['Moc'].astype(float)

        return X_


class RenameColumns(BaseEstimator, TransformerMixin):
    def __init__(self, rename_dict={'Price': 'Cena', 'Features': 'Wyposażenie',
                                    'Description': 'Opis', 'City': 'Miasto',
                                    'Latitude': 'Sz. geograficzna',
                                    'Longitude': 'Dł. geograficzna'}):
        self.rename_dict = rename_dict

    def fit(self, X):
        return self

    def transform(self, X):
        return X.rename(columns=self.rename_dict)


class RemoveColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_delete=['id', 'Kategoria', 'Wersja', 'Spalanie W Mieście', 'Emisja CO2',
                                          'Wyposażenie', 'Opis', 'Miasto', 'Generacja', 'Okres gwarancji producenta',
                                          'Spalanie W Cyklu Mieszanym', 'Spalanie Poza Miastem', 'Tuning',
                                          'Numer rejestracyjny pojazdu', 'Filtr cząstek stałych',
                                          'Opłata początkowa', 'Miesięczna rata', 'Liczba pozostałych rat',
                                          'Wartość wykupu', 'lub do (przebieg km)', 'Gwarancja dealerska (w cenie)',
                                          'Kierownica po prawej (Anglik)', 'Homologacja ciężarowa',
                                          'Zarejestrowany jako zabytek', 'VIN', 'Pierwsza rejestracja'],
                 delete_models=False):
        self.columns_to_delete = columns_to_delete
        self.delete_models = delete_models

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()

        X_.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1, inplace=True)
        X_.drop(self.columns_to_delete, axis=1, inplace=True)
        if self.delete_models:
            X_.drop('Model pojazdu', axis=1, inplace=True)

        return X_


class NanToBinary(BaseEstimator, TransformerMixin):
    def __init__(self, columns=['Możliwość finansowania', 'Zarejestrowany w Polsce', 'Pierwszy właściciel',
                                'Bezwypadkowy', 'Serwisowany w ASO', 'Faktura VAT', 'Leasing', 'VAT marża', 'Uszkodzony']):
        self.columns = columns

    def fit(self, X):
        return self

    def transform(self, X):
        X_ = X.copy()

        for column in self.columns:
            X_.loc[(X_[column] == 'Tak'), column] = True
            X_.loc[(X_[column].isna()), column] = False

        return X_


class RemoveRowsWithNan(BaseEstimator, TransformerMixin):
    def __init__(self, columns=['Przebieg', 'Pojemność skokowa', 'Moc']):
        self.columns = columns

    def fit(self, X):
        return self

    def transform(self, X):
        return X.dropna(subset=self.columns)

# Transformator zamieniające rzadkie marki i modele samochodu na "Inne"


class RareToOther(BaseEstimator, TransformerMixin):
    def __init__(self, brand_threshold=0.001, model_threshold=0.0001, country_threshold=0.001):
        self.brand_threshold = brand_threshold
        self.model_threshold = model_threshold
        self.country_threshold = country_threshold
        self.brands = None
        self.models = None
        self.countries = None
        self.brand_idx = None
        self.model_idx = None
        self.country_idx = None

    def fit(self, X, y=None):
        self.brand_idx = np.argmax(np.bincount(
            np.where(np.isin(X, ['Opel', 'Audi', 'BMW', 'Ford', 'Volkswagen']))[1]))
        brand_occurrences = np.array(
            np.unique(X[:, self.brand_idx], return_counts=True))
        self.brands = brand_occurrences[:, brand_occurrences[1,
                                                             :] / len(X) > self.brand_threshold][0]

        self.model_idx = np.argmax(np.bincount(
            np.where(np.isin(X, ['Astra', 'Seria 3', 'A4', 'Seria 5', 'Golf']))[1]))
        model_occurrences = np.array(
            np.unique(X[:, self.model_idx], return_counts=True))
        self.models = model_occurrences[:, model_occurrences[1,
                                                             :] / len(X) > self.model_threshold][0]

        self.country_idx = np.argmax(np.bincount(
            np.where(np.isin(X, ['Polska', 'Niemcy']))[1]))
        country_occurences = np.array(
            np.unique(X[:, self.country_idx], return_counts=True)
        )
        self.countries = country_occurences[:, country_occurences[1, :] / len(
            X) > self.country_threshold][0]

        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        X_[:, self.brand_idx][~np.isin(
            X[:, self.brand_idx], self.brands)] = 'Other'
        X_[:, self.model_idx][~np.isin(
            X[:, self.model_idx], self.models)] = 'Other'
        X_[:, self.country_idx][~np.isin(
            X[:, self.country_idx], self.countries)] = 'Other'
        return X_


class RemoveOutliers(BaseEstimator, TransformerMixin):
    def __init__(self, columns=['Cena', 'Przebieg', 'Moc', 'Pojemność skokowa'], percent_to_remove=0.01):
        self.columns = columns
        self.percent_to_remove = percent_to_remove

    def fit(self, X):
        return self

    def transform(self, X):
        for column in self.columns:
            X = X.sort_values(by=column).iloc[int(
                len(X) * self.percent_to_remove): -int(len(X) * self.percent_to_remove)]
        return X
