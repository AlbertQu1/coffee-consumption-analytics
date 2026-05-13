import pandas as pd

# URL del Google Sheet principal
url_principal = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTdOWbzsFlVlSTBguq9_cjLXzvDO-uUqnIY4zaex27J4biHRk2t5u7aCHShyaCVmKhtJ12XLDQ8Nu2n/pub?gid=0&single=true&output=csv"

# Leer el dataframe principal
df = pd.read_csv(url_principal)

# datos iniciales
costo_cafetera = 12239
precio_cafeteria = {
    "Elena": 70,
    "Chavelete": 60,
    "Coffee King": 50,
    "Starbucks": 58,
    "Quentin": 55,
    "Quentin 2": 70,
    "otro": 40,
}
precio_por_taza_cafeteria = sum(precio_cafeteria.values()) / len(precio_cafeteria)

# imprimir indicando que son kg
cafe_gastado = df["gramos"].sum() / 1000

# tazas de  Cafe
columnas_tazas = [
    "espresso",
    "quad",
    "6oz",
    "8oz",
    "10oz",
    "12oz",
    "14oz",
    "16oz",
    "18oz",
]
total_tazas_tipo = df[columnas_tazas].fillna(0).sum()
total_tazas = df[columnas_tazas].fillna(0).sum().sum()

# costo de cafe
costo_total_cafe = df["costo"].sum()
costo_unitario_taza = costo_total_cafe / total_tazas if total_tazas > 0 else 0

# ahorro en cafeteria
ahorro_vs_cafeteria = precio_por_taza_cafeteria - costo_unitario_taza

# punto de equilibrio
punto_de_equilibrio = costo_cafetera / ahorro_vs_cafeteria

# ahorro
ahorro_total = ahorro_vs_cafeteria * total_tazas

print(f"\n📊 Análisis general:")
print(f"- Precio promedio en cafeteria: ${precio_por_taza_cafeteria:.2f} MXN")
print(f"- Costo promedio en casa: ${costo_unitario_taza:.2f} MXN")
print(f"- Cafe consumido: {cafe_gastado} kg")
print(f"- Ahorro por taza: {ahorro_vs_cafeteria:.2f} MXN")
print(f"- Tazas consumidas {total_tazas}")
print(f"- Tazas para el punto de equilibrio: {punto_de_equilibrio:.0f}")
print(f"- Ahorro total al momento: ${ahorro_total:.2f}")

print("\n🕒 Tiempo para recuperar la inversión:")
for tazas_dia in [1, 2, 3, 4]:
    dias = punto_de_equilibrio / tazas_dia
    meses = dias / 30
    print(f"- {tazas_dia} tazas por día: {dias:.0f} días (~{meses:.1f} meses)")

print(f"\n💰 Costo promedio del cafe:")
cafeterias = df["Cafeteria"].unique()
for nombre in cafeterias:
    promedio = df[df["Cafeteria"] == nombre]["costo"].mean()
    print(f"- {nombre}: ${promedio:.2f} MXN")

print(f"\n☕ Tipos de tazas:")
for tipo, cantidad in total_tazas_tipo.items():
    print(f"- {tipo}: {cantidad}")
