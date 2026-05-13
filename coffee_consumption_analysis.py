#datos iniciales
costo_maquina= 12239

#costo en diferentes cafeterias
precios_cafeterias= {
    'Elena': 70,
    'Chavelete': 60,
    'king Coffee': 50,
    'Starbucks': 58,
    'Quentin': 55,
    'Quentin': 70
}

#promedio de los precios en las diferentes cafeterias
precio_taza_cafeteria = sum(precios_cafeterias.values()) / len(precios_cafeterias)

#cafes comprados
compras_cafe = [
    {
        'gramos': 500,
        'costo': 120,
        'tazas': {
            'espresso': 20,
            'quad': 1,
            '8oz': 1
        }
    },
    {
        'gramos': 500,
        'costo': 192.5,
        'tazas': {
            'espresso': 24,
            'quad': 2,
            '8oz': 2
        }
    },
    {
        'gramos': 100,
        'costo': 50,
        'tazas': {
            'espresso': 6
        }
    },
    {
        'gramos': 500,
        'costo': 375,
        'tazas': {
            'espresso': 19,
            '6oz': 2,
            '8oz': 2,
            '10oz': 2,
            '12oz': 1,
            '14oz': 2
        }
    }
]

#directorios para llenar
total_cafe_gastado= 0
total_tazas= 0
tipo_tazas= {}
costo_tipo= {}

#utilizamos un bucle apra recorrer cada compra
for compra in compras_cafe:
    costo= compra['costo']
    tazas_dic = compra['tazas']

    tazas_compra= sum(tazas_dic.values())
    total_cafe_gastado += costo
    total_tazas += tazas_compra

    for tipo, cantidad in tazas_dic.items():
        costo_proporcional= (cantidad/ tazas_compra) * costo
        tipo_tazas[tipo]= tipo_tazas.get(tipo,0)+ cantidad
        costo_tipo[tipo]= costo_tipo.get(tipo,0) + costo_proporcional

#calculos generales
inversion_total= costo_maquina + total_cafe_gastado
costo_promedio_taza= total_cafe_gastado / total_tazas
ahorro_por_taza= precio_taza_cafeteria - costo_promedio_taza
ahorro_acumulado= ahorro_por_taza * total_tazas
punto_de_equilibrio= inversion_total / ahorro_por_taza

#resultados generales
print(f'\n📊 Análisis general:')
print(f'- Precio promedio en cafeterías: ${precio_taza_cafeteria:.2f} MXN')
print(f'- Costo promedio por taza en casa: ${costo_promedio_taza:.2f} MXN')
print(f'- Ahorro por taza vs cafetería: ${ahorro_por_taza:.2f} MXN')
print(f'- Tazas para el punto de equilibrio: {punto_de_equilibrio:.0f}')
print(f'- Total de tazas consumidas al momento {total_tazas}')

print('\n🕒 Tiempo para recuperar la inversión:')
for tazas_dia in [1,2,3,4]:
    dias= punto_de_equilibrio /tazas_dia
    meses= dias/30
    print(f'- {tazas_dia} tazas por día: {dias:.0f} días (~{meses:.1f} meses)')

#resultado por tipo de tazas
print(f'\n📌 Detalle por tipo de taza:')
for tipo in tipo_tazas:
    tazas= tipo_tazas[tipo]
    costo= costo_tipo[tipo]
    costo_promedio= costo / tazas
    ahorro = (precio_taza_cafeteria - costo_promedio)* tazas
    print(f'- {tipo.capitalize()}: {tazas} tazas, ${costo_promedio:.2f} promedio, ahorro total ${ahorro:.2f}')

# --- Nueva sección: Gráfica de barras por tipo de taza ---
import matplotlib.pyplot as plt

tipos = list(tipo_tazas.keys())
cantidades = list(tipo_tazas.values())

plt.figure(figsize=(8,5))
plt.bar(tipos, cantidades, color='saddlebrown')
plt.xlabel('Tipo de taza')
plt.ylabel('Cantidad consumida')
plt.title('Consumo de café por tipo de taza')
plt.tight_layout()
plt.show()
