import logging
from typing import Dict, Any, List
from datetime import datetime
from src.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class GAAPNormalizerAgent(BaseAgent):
    """
    Agente responsable de procesar los 'company facts' de la SEC.
    V20 FÍSICA CONTABLE + SECTOR SEGUROS/BANCOS (FIX PGR):
    Añade etiquetas de deuda atípicas (DebtInstrumentCarryingAmount, NotesPayable)
    utilizadas por aseguradoras y financieras para evitar ceros en Deuda a Largo Plazo.
    """

    def __init__(self):
        super().__init__(
            agent_name="GAAPNormalizerAgent",
            prompt_file="gaap_normalizer.xml",
            temperature=0.0
        )

    def normalize(self, raw_sec_data: Dict[str, Any]) -> Dict[str, Any]:
        ticker = raw_sec_data.get("ticker", "UNKNOWN")
        logger.info(f"[{self.agent_name}] Aplicando Física Contable y Diccionario de Seguros/Bancos para {ticker}...")

        facts = raw_sec_data.get("raw_facts", {}).get("facts", {})

        target_tags = {
            # --- ESTADO DE RESULTADOS ---
            "Ingresos": ["Revenues", "SalesRevenueNet", "SalesRevenueGoodsNet", "Revenue", "PremiumsEarnedNet"],
            "Ingresos financieros": ["FinancialServicesRevenue", "InterestIncomeOperating"],
            "Ingresos de la division de seguros": ["InsurancePremiums", "InsuranceCommissionsAndFees", "PremiumsEarnedNet"],
            "Ganancia (perdida) en la venta de activos (Rev)": ["GainLossOnSaleOfAssets", "GainLossOnDispositionOfAssets"],
            "Ganancia (perdida) en venta de inversion (Rev)": ["GainLossOnSaleOfInvestments"],
            "Intereses e ingresos de inversiones": ["InvestmentIncomeInterestAndDividend", "InvestmentIncomeInterest", "InvestmentIncomeNet", "NetInvestmentIncome"],
            "Otros ingresos": ["OtherIncome", "OtherOperatingIncome", "NonoperatingIncomeExpense"],
            "Ingresos totales": ["Revenues", "SalesRevenueNet", "TotalRevenuesAndOtherIncome", "RevenueFromContractWithCustomerExcludingAssessedTax", "RevenuesNetOfInterestExpense", "OperatingLeasesIncomeStatementLeaseRevenue", "Revenue", "HomebuildingRevenues", "HomebuildingAndLandSalesRevenues", "RealEstateRevenue", "SalesRevenueServicesNet", "RevenueFromContractWithCustomerIncludingAssessedTax"],
            
            "Coste de los bienes vendidos": ["CostOfGoodsAndServicesSold", "CostOfRevenue", "CostOfGoodsSold", "CostOfServices", "CostOfSales", "HomebuildingCostOfRevenues", "HomebuildingAndLandSalesCostOfRevenues", "CostOfHomebuildingRevenues", "CostOfRealEstateSales", "CostOfHomebuildingAndLandSales", "HomebuildingCostOfSales", "CostOfGoodsSoldExcludingDepreciationAndAmortization", "CostOfGoodsSoldDepreciationAndAmortization"],
            
            "Gastos financieros operativos": ["InterestExpenseOperating", "OperatingFinancialExpenses"],
            "Gastos operativos de la division de seguros": ["InsuranceBenefitsAndClaimsPaid", "PolicyholderBenefitsAndClaimsIncurredNet", "LossesAndLossAdjustmentExpenses"],
            "Gastos por intereses - Division de finanzas": ["InterestExpenseDebt", "InterestExpense"],
            
            "Beneficio bruto": ["GrossProfit", "GrossProfitMargin", "HomebuildingGrossMargin", "HomebuildingGrossProfit"],
            
            "Gastos de venta generales y administrativos": ["SellingGeneralAndAdministrativeExpense", "SellingAndMarketingExpense", "GeneralAndAdministrativeExpense", "SellingExpense", "HomebuildingSellingGeneralAndAdministrativeExpense"],
            "Gastos de exploracion / perforacion": ["ExplorationExpense", "DryHoleCosts"],
            "Provision para deudas incobrables": ["ProvisionForDoubtfulAccounts", "ProvisionForCreditLosses", "AssetImpairmentCharges"],
            "Gastos de pre-apertura": ["PreOpeningCosts", "PreopeningExpense"],
            "Gastos de I + D": ["ResearchAndDevelopmentExpense", "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost"],
            "Deterioro del petroleo, gas y propiedades minerales": ["ImpairmentOfOilAndGasProperties"],
            "Otros gastos operacionales": ["OtherOperatingIncomeExpenseNet", "OtherOperatingExpenses", "OperatingCostsAndExpenses"],
            "Gastos operativos totales": ["OperatingExpenses", "CostsAndExpenses", "TotalOperatingExpenses"],
            "Beneficio operativo": ["OperatingIncomeLoss", "OperatingIncome"],
            "Gastos por intereses": ["InterestExpense", "InterestExpenseNet", "InterestExpenseDebt", "InterestCostsIncurred"],
            "Ingresos por intereses e inversiones": ["InvestmentIncomeInterestAndDividend", "InvestmentIncomeNet"],
            "Ingresos (perdidas) sobre capital invertido.": ["IncomeLossFromEquityMethodInvestments"],
            "Ganancias (perdidas) cambiarias": ["ForeignCurrencyTransactionGainLossBeforeTax", "ForeignCurrencyTransactionGainLossRealized"],
            "Otros ingresos (gastos) no operativos": ["OtherNonoperatingIncomeExpense", "NonoperatingIncomeExpense", "OtherIncomeExpense"],
            "EBT excl. Articulos inusuales": ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments", "IncomeBeforeIncomeTaxes"],
            "Cargos de fusion y reestructuracion": ["RestructuringCharges", "BusinessCombinationIntegrationRelatedCosts", "RestructuringAndRelatedCost"],
            "Deterioro del fondo de comercio": ["GoodwillImpairmentLoss"],
            "Gain (Loss) On Sale Of Investments": ["GainLossOnSaleOfInvestments"],
            "Ganancia (perdida) en la venta de activos": ["GainLossOnSaleOfPropertyPlantEquipment", "GainLossOnDispositionOfAssets"],
            "Devaluacion de activos": ["AssetImpairmentCharges"],
            "Gastos de I + D en proceso": ["ResearchAndDevelopmentInProcess", "AcquiredInProcessResearchAndDevelopmentExpense"],
            "Liquidaciones de seguros": ["ProceedsFromInsuranceSettlement"],
            "Acuerdos legales": ["LitigationSettlementExpense", "LitigationSettlement"],
            "Otros articulos inusuales": ["UnusualOrInfrequentItemNetOfTax"],
            "EBT incl. Articulos extraordinarios": ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest", "IncomeLossFromContinuingOperationsBeforeIncomeTaxesDomestic"],
            "Gastos de impuestos": ["IncomeTaxExpenseBenefit", "IncomeTaxExpenseBenefitContinuingOperations", "CurrentIncomeTaxExpenseBenefit"],
            "Beneficios por operaciones continuadas": ["IncomeLossFromContinuingOperations"],
            "Beneficios por operaciones discontinuadas": ["IncomeLossFromDiscontinuedOperationsNetOfTax"],
            "Articulo extraordinario y cambio contable": ["ExtraordinaryItemNetOfTax"],
            "Beneficio neto de la empresa": ["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic", "IncomeLossFromContinuingOperations"],
            "Dividendo preferente y otros ajustes": ["PreferredStockDividends", "PreferredStockDividendsAndOtherAdjustments"],
            "Beneficio neto a acciones comunes incluidos extradordinarios": ["NetIncomeLossAvailableToCommonStockholdersBasic"],
            "Beneficio neto a acciones comunes excluidos extradordinarios": ["NetIncomeLossAvailableToCommonStockholdersDiluted"],
            "BPA diluido sin extraordinarios": ["EarningsPerShareDiluted"],
            
            "Promedio ponderado de acciones diluidas en circulacion": ["WeightedAverageNumberOfDilutedSharesOutstanding", "WeightedAverageNumberOfShareOutstandingBasicAndDiluted", "WeightedAverageNumberOfSharesOutstandingBasic", "EntityCommonStockSharesOutstanding"],
            "Promedio ponderado de acciones basicas en circulacion": ["WeightedAverageNumberOfSharesOutstandingBasic", "WeightedAverageNumberOfShareOutstandingBasicAndDiluted", "EntityCommonStockSharesOutstanding"],
            "Total de acciones fuera. en la fecha de presentacion": ["EntityCommonStockSharesOutstanding", "CommonStockSharesOutstanding", "CommonStockSharesIssued", "WeightedAverageNumberOfSharesOutstandingBasic"],
            
            "Dividendo por accion": ["CommonStockDividendsPerShareDeclared", "DividendsPerShare", "CommonStockDividendsPerShareCashPaid"],
            "Dividendo especial por accion": ["SpecialDividend", "DividendsPerShareSpecial"],
            "BPA basico": ["EarningsPerShareBasic"],
            
            "FFO": ["FundsFromOperations", "FundsFromOperationsAvailableToCommonStockholders"],
            "EBITDA": ["EBITDA", "NetIncomeLossBeforeInterestTaxesDepreciationAndAmortization"],
            "EBITDAR": ["EBITDAR"],
            "Gasto en I + D ": ["ResearchAndDevelopmentExpense"],
            "Gastos de venta y marketing": ["SellingAndMarketingExpense", "MarketingExpense"],
            "Gastos generales y administrativos": ["GeneralAndAdministrativeExpense"],
            "Efectivo distribuible por accion (diluido)": ["DistributableCashFlowPerShare"],
            "Distribuciones anualizadas por unidad": ["AnnualizedDistributionsPerUnit"],

            # --- BALANCE GENERAL ---
            "Efectivo y equivalentes": ["CashAndCashEquivalentsAtCarryingValue", "Cash", "CashAndDueFromBanks"],
            "Inversiones a corto plazo": ["ShortTermInvestments", "AvailableForSaleSecuritiesCurrent"],
            "Activos financieros para vender": ["AvailableForSaleSecuritiesCurrent", "AvailableForSaleSecurities", "FixedMaturitiesAvailableForSale"],
            "Efectivo total e inversiones a corto plazo": ["CashCashEquivalentsAndShortTermInvestments", "CashAndShortTermInvestments"],
            "Cuentas por cobrar": ["AccountsReceivableNetCurrent", "ReceivablesNetCurrent", "AccountsNotesAndLoansReceivableNetCurrent", "PremiumsReceivableAtCarryingValue"],
            "Otros por cobrar": ["OtherReceivablesNetCurrent", "OtherReceivables"],
            "Notas por cobrar": ["NotesReceivableNet"],
            "Total de cuentas por cobrar": ["AccountsReceivableNetCurrent", "ReceivablesNetCurrent"],
            "Inventario": ["InventoryNet", "InventoryGross", "Inventories", "RealEstateInventory"],
            "Gastos pagados por anticipado": ["PrepaidExpenseCurrent", "PrepaidExpenseAndOtherAssetsCurrent", "PrepaidExpense"],
            "Prestamos y arrendamientos de la division de finanzas al corriente": ["FinanceLeasePrincipalPaymentsDue"],
            "Division de Finanzas Otros Activos Circulantes": ["OtherAssetsCurrent"],
            "Prestamos mantenidos para la venta": ["LoansHeldForSaleNet"],
            "Activos por impuestos diferidos Corrientes": ["DeferredTaxAssetsNetCurrent"],
            "Efectivo restringido": ["RestrictedCashAndCashEquivalentsAtCarryingValue", "RestrictedCash"],
            "Otro activo corriente": ["OtherAssetsCurrent", "OtherCurrentAssets"],
            "Total de activo corriente": ["AssetsCurrent"],
            "Inmobilizado material bruto": ["PropertyPlantAndEquipmentGross"],
            "Depreciacion acumulada": ["AccumulatedDepreciationDepletionAndAmortizationPropertyPlantAndEquipment"],
            "Inmobilizado material neto": ["PropertyPlantAndEquipmentNet"],
            "Inversiones a largo plazo": ["LongTermInvestments", "AvailableForSaleSecuritiesNoncurrent"],
            "Fondo de comercio": ["Goodwill", "GoodwillAndIntangibleAssetsNet"],
            "Otros intangibles": ["IntangibleAssetsNetExcludingGoodwill", "IntangibleAssetsNet"],
            "Division de Finanzas Prestamos y Arrendamientos a Largo Plazo": ["FinanceLeaseLiabilityNoncurrent"],
            "Division financiera Otros activos a largo plazo": ["OtherAssetsNoncurrent"],
            "Cuentas por cobrar a largo plazo": ["AccountsReceivableNetNoncurrent"],
            "Prestamos por cobrar a largo plazo": ["LoansReceivableNet", "LoansAndLeasesReceivableNetOfAllowance"],
            "Activos por impuestos diferidos a largo plazo": ["DeferredTaxAssetsNetNoncurrent", "DeferredTaxAssetsNet"],
            "Cargos diferidos a largo plazo": ["DeferredCharges"],
            "Otros activos a largo plazo": ["OtherAssetsNoncurrent", "OtherNoncurrentAssets"],
            "Activo total": ["Assets"],

            "Cuentas por pagar": ["AccountsPayableCurrent", "AccountsPayableAndAccruedLiabilitiesCurrent", "AccountsPayable"],
            "Gastos devengados": ["AccruedLiabilitiesCurrent", "AccruedLiabilities"],
            
            # Ampliación de Deuda a Corto Plazo
            "Prestamos de corto plazo": ["ShortTermBorrowings", "DebtCurrent", "CommercialPaper", "NotesPayableCurrent", "LinesOfCreditCurrent", "ShortTermDebt"],
            "Porcion corriente de la deuda a largo plazo": ["LongTermDebtCurrent", "LongTermDebtAndCapitalLeaseObligationsCurrent"],
            "Porcion corriente de las obligaciones de arrendamiento financiero": ["FinanceLeaseLiabilityCurrent", "CapitalLeaseObligationsCurrent"],
            "Deuda corriente de la Division de Finanzas": ["DebtCurrent"],
            "Otros Pasivos Corrientes de la Division de Finanzas": ["OtherLiabilitiesCurrent"],
            "Impuestos sobre la renta actuales por pagar": ["IncomeTaxesPayable", "AccruedIncomeTaxesCurrent"],
            "Unearned Revenue Current": ["ContractWithCustomerLiability", "DeferredRevenueCurrent", "DeferredRevenue", "BillingsInExcessOfCostCurrent", "UnearnedPremiums"],
            "Pasivo por impuestos diferidos Corriente": ["DeferredTaxLiabilitiesCurrent"],
            "Otros pasivos corrientes": ["OtherLiabilitiesCurrent", "OtherCurrentLiabilities"],
            "Total pasivo corriente": ["LiabilitiesCurrent"],
            
            # Ampliación Crítica de Deuda a Largo Plazo (PGR y Aseguradoras)
            "Deuda a largo plazo": ["LongTermDebt", "LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations", "DebtInstrumentCarryingAmount", "NotesPayable", "LongTermNotesPayable", "SeniorNotes"],
            
            "Arrendamientos de capitales": ["FinanceLeaseLiabilityNoncurrent", "OperatingLeaseLiabilityNoncurrent", "CapitalLeaseObligations"],
            "Deuda No Corriente de la Division de Finanzas": ["LongTermDebt"],
            "Otro Pasivo No Corriente de la Division de Finanzas": ["OtherLiabilitiesNoncurrent"],
            "Ingresos no devengados no corrientes": ["DeferredRevenueNoncurrent", "ContractWithCustomerLiabilityNoncurrent"],
            "Pension y otros beneficios posteriores a la jubilacion": ["PensionAndOtherPostretirementDefinedBenefitPlansLiabilitiesNoncurrent"],
            "Pasivo por impuesto diferido no corriente": ["DeferredTaxLiabilitiesNoncurrent", "DeferredTaxLiabilities"],
            "Otro pasivo no corrientes": ["OtherLiabilitiesNoncurrent", "OtherNoncurrentLiabilities"],
            "Pasivo Total": ["Liabilities"],

            "Acciones preferentes reembolsables": ["RedeemablePreferredStockValue", "TemporaryEquityValue"],
            "Acciones preferentes no reembolsables": ["NonredeemablePreferredStockValue"],
            "Acciones preferentes convertibles": ["ConvertiblePreferredStockValue"],
            "Total preferentes": ["PreferredStockValue"],
            "Acciones comunes": ["CommonStockValue", "CommonStockValueOutstanding"],
            "Prima de suscripcion": ["AdditionalPaidInCapital", "AdditionalPaidInCapitalCommonStock"],
            "Beneficio no distribuido": ["RetainedEarningsAccumulatedDeficit", "RetainedEarnings"],
            "Autocartera": ["TreasuryStockValue"],
            "Resultado integral y otros": ["AccumulatedOtherComprehensiveIncomeLossNetOfTax"],
            "Patrimonio neto comun total": ["StockholdersEquity"],
            "Intereses minoritario": ["MinorityInterest", "NoncontrollingInterestInConsolidatedEntity"],
            "Fondos propios totales": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest", "PartnersCapital"],
            "Pasivo total y patrimonio neto": ["LiabilitiesAndStockholdersEquity"],
            
            # --- DATOS ADICIONALES Y FLUJO ---
            "Inversiones por metodo de participacion": ["EquityMethodInvestments"],
            "Terrenos": ["Land"],
            "Edificios": ["BuildingsAndImprovementsGross"],
            "Construccion en progreso": ["ConstructionInProgressGross"],
            "Empleados a tiempo completo": ["EntityNumberOfEmployees", "NumberOfEmployees"],

            "Beneficio netos": ["NetIncomeLoss", "ProfitLoss"],
            "Amortizacion de fondos de comercio y activos intangibles": ["AmortizationOfIntangibleAssets"],
            "Depreciacion y amortizacion total": ["DepreciationAndAmortization"],
            "Amortizacion de cargos diferidos": ["AmortizationOfDeferredCharges"],
            "Interes minoritario en las ganancias": ["NetIncomeLossAttributableToNoncontrollingInterest"],
            "(Ganancia) Perdida por venta de activos": ["GainLossOnSaleOfAssets", "GainLossOnSaleOfPropertyPlantEquipment"],
            "(Ganancia) Perdida por venta de inversiones": ["GainLossOnSaleOfInvestments"],
            "Deterioro de actiovs y costes de reestructuracion": ["RestructuringAndImpairmentCharges"],
            "Provision para perdidas crediticias": ["ProvisionForCreditLosses"],
            "(Ingresos) Perdida en inversiones de capital": ["IncomeLossFromEquityMethodInvestments"],
            "Compensacion de stock options": ["ShareBasedCompensation", "AllocatedShareBasedCompensationExpense"],
            "Beneficio fiscal de las opciones sobre acciones": ["TaxBenefitFromShareBasedCompensation"],
            "Provision y cancelacion de deudas incobrables": ["ProvisionForDoubtfulAccounts"],
            "Efectivo neto de operaciones discontinuadas": ["NetCashProvidedByUsedInOperatingActivitiesDiscontinuedOperations"],
            "Otras actividades operativas": ["OtherOperatingActivitiesCashFlowStatement"],
            "Cambio en cuentas por cobrar": ["IncreaseDecreaseInAccountsReceivable"],
            "Cambio en inventarios": ["IncreaseDecreaseInInventories"],
            "Cambio en cuentas por pagar": ["IncreaseDecreaseInAccountsPayable"],
            "Cambio en los ingresos no devengados": ["IncreaseDecreaseInDeferredRevenue"],
            "Cambio de impuestos sobre la renta": ["IncreaseDecreaseInIncomeTaxes"],
            "Efectivo de Operaciones": ["NetCashProvidedByOperatingActivities", "NetCashProvidedByUsedInOperatingActivities", "NetCashProvidedByOperatingActivitiesContinuingOperations"],
            "Gastos de capital": ["PaymentsForPropertyPlantAndEquipment", "PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
            "Venta de inmobilizado material": ["ProceedsFromSaleOfPropertyPlantAndEquipment"],
            "Adquisiciones con efectivo": ["PaymentsToAcquireBusinessesNetOfCashAcquired"],
            "Desinversiones": ["ProceedsFromDivestitures"],
            "Inversion en valores negociables y de renta variable": ["PaymentsToAcquireMarketableSecurities", "PaymentsToAcquireAvailableForSaleSecurities"],
            "Efectivo de la inversion": ["NetCashProvidedByUsedInInvestingActivities", "NetCashProvidedByInvestingActivities"],
            "Deuda total emitida": ["ProceedsFromIssuanceOfLongTermDebt", "ProceedsFromIssuanceOfDebt"],
            "Total de la deuda reembolsada": ["RepaymentsOfLongTermDebt", "RepaymentsOfDebt"],
            "Emision de acciones ordinarias": ["ProceedsFromIssuanceOfCommonStock"],
            "Recompra de acciones comunes": ["PaymentsForRepurchaseOfCommonStock", "PaymentsForRepurchaseOfEquity", "PaymentsForRepurchaseOfSecurities", "StockRepurchasedAndRetiredDuringPeriodValue", "StockRepurchasedDuringPeriodValue"],
            "Emision de acciones preferentes": ["ProceedsFromIssuanceOfPreferredStock"],
            "Dividendos comunes pagados": ["PaymentsOfDividendsCommonStock", "PaymentsOfDividends", "DividendsCash", "PaymentsOfOrdinaryDividends"],
            "Dividendos preferenciales pagados": ["PaymentsOfDividendsPreferredStockAndPreferenceStock"],
            "Dividendos de acciones comunes y preferentes pagados": ["PaymentsOfDividends", "PaymentsOfDividendsCommonStock", "Dividends", "PaymentsOfDividendsPreferredStockAndPreferenceStock", "PaymentsOfOrdinaryDividends", "DividendsCash"],
            "Efectivo de Financiamiento": ["NetCashProvidedByUsedInFinancingActivities", "NetCashProvidedByFinancingActivities"],
            "Ajustes del tipo de cambio de divisas": ["EffectOfExchangeRateOnCashAndCashEquivalents"],
            "Cambio neto en efectivo": ["CashAndCashEquivalentsPeriodIncreaseDecrease", "CashPeriodIncreaseDecrease"],
            "Intereses en efectivo pagados": ["InterestPaidNet", "InterestPaid"],
            "Impuestos en efectivo pagados": ["IncomeTaxesPaidNet", "IncomeTaxesPaid"]
        }

        normalized_data = {"ticker": ticker, "series": {}}
        all_years = set()

        for standard_key, possible_tags in target_tags.items():
            series, years = self._extract_fact_series(facts, possible_tags)
            normalized_data["series"][standard_key] = series
            all_years.update(years)

        sorted_years = sorted([y for y in all_years if len(y) == 4 and y.isdigit()])[-12:]
        normalized_data["years"] = sorted_years

        aligned_series = {}
        for key, series_dict in normalized_data["series"].items():
            aligned_series[key] = [series_dict.get(y, 0.0) for y in sorted_years]

        # =========================================================================
        # MOTOR ALGEBRAICO MULTI-PASADA (SUDOKU FINANCIERO)
        # =========================================================================
        def get_v(k, i): return aligned_series.get(k, [0.0]*len(sorted_years))[i]
        def set_v(k, i, val): 
            if k not in aligned_series: aligned_series[k] = [0.0]*len(sorted_years)
            if aligned_series[k][i] == 0.0 and val != 0.0:
                aligned_series[k][i] = val

        for i in range(len(sorted_years)):
            # 3 PASADAS para romper dependencias circulares
            for _ in range(3):
                # 1. Acciones
                if get_v("Promedio ponderado de acciones basicas en circulacion", i) == 0.0:
                    set_v("Promedio ponderado de acciones basicas en circulacion", i, get_v("Total de acciones fuera. en la fecha de presentacion", i))
                if get_v("Promedio ponderado de acciones diluidas en circulacion", i) == 0.0:
                    set_v("Promedio ponderado de acciones diluidas en circulacion", i, get_v("Promedio ponderado de acciones basicas en circulacion", i))

                # 2. Beneficio Operativo
                if get_v("Beneficio operativo", i) == 0.0 and get_v("EBT incl. Articulos extraordinarios", i) != 0.0:
                    set_v("Beneficio operativo", i, get_v("EBT incl. Articulos extraordinarios", i) + abs(get_v("Gastos por intereses", i)))

                # 3. Gastos VGA y OpEx
                if get_v("Gastos de venta generales y administrativos", i) == 0.0:
                    set_v("Gastos de venta generales y administrativos", i, get_v("Gastos de venta y marketing", i) + get_v("Gastos generales y administrativos", i))
                
                if get_v("Gastos operativos totales", i) == 0.0:
                    calc_opex = get_v("Gastos de venta generales y administrativos", i) + get_v("Gastos de I + D", i) + get_v("Depreciacion y amortizacion", i) + get_v("Otros gastos operacionales", i)
                    if calc_opex > 0: set_v("Gastos operativos totales", i, calc_opex)
                    elif get_v("Beneficio bruto", i) != 0.0 and get_v("Beneficio operativo", i) != 0.0:
                        set_v("Gastos operativos totales", i, get_v("Beneficio bruto", i) - get_v("Beneficio operativo", i))

                # 4. PUENTE ALGEBRAICO
                if get_v("Beneficio bruto", i) == 0.0 and get_v("Coste de los bienes vendidos", i) == 0.0:
                    if get_v("Ingresos totales", i) != 0.0 and get_v("Beneficio operativo", i) != 0.0 and get_v("Gastos operativos totales", i) != 0.0:
                        cogs_calculado = get_v("Ingresos totales", i) - get_v("Beneficio operativo", i) - get_v("Gastos operativos totales", i)
                        if cogs_calculado > 0:
                            set_v("Coste de los bienes vendidos", i, cogs_calculado)

                # 5. Ecuaciones Normales de Resultados
                if get_v("Coste de los bienes vendidos", i) == 0.0 and get_v("Ingresos totales", i) != 0.0 and get_v("Beneficio bruto", i) != 0.0:
                    set_v("Coste de los bienes vendidos", i, get_v("Ingresos totales", i) - get_v("Beneficio bruto", i))
                
                if get_v("Beneficio bruto", i) == 0.0:
                    if get_v("Ingresos totales", i) != 0.0 and get_v("Coste de los bienes vendidos", i) != 0.0:
                        set_v("Beneficio bruto", i, get_v("Ingresos totales", i) - get_v("Coste de los bienes vendidos", i))
                    elif get_v("Beneficio operativo", i) != 0.0 and get_v("Gastos operativos totales", i) != 0.0:
                        set_v("Beneficio bruto", i, get_v("Beneficio operativo", i) + get_v("Gastos operativos totales", i))

                if get_v("Beneficio operativo", i) == 0.0 and get_v("Beneficio bruto", i) != 0.0 and get_v("Gastos operativos totales", i) != 0.0:
                    set_v("Beneficio operativo", i, get_v("Beneficio bruto", i) - get_v("Gastos operativos totales", i))

                if get_v("Gastos de impuestos", i) == 0.0 and get_v("EBT incl. Articulos extraordinarios", i) != 0.0:
                    set_v("Gastos de impuestos", i, get_v("EBT incl. Articulos extraordinarios", i) - get_v("Beneficio neto de la empresa", i))


                # 6. Identidades de Balance y Flujos
                if get_v("Activo total", i) == 0.0 and get_v("Pasivo Total", i) != 0.0:
                    set_v("Activo total", i, get_v("Pasivo Total", i) + get_v("Fondos propios totales", i))
                if get_v("Pasivo Total", i) == 0.0 and get_v("Activo total", i) != 0.0:
                    set_v("Pasivo Total", i, get_v("Activo total", i) - get_v("Fondos propios totales", i))
                if get_v("Fondos propios totales", i) == 0.0 and get_v("Activo total", i) != 0.0:
                    set_v("Fondos propios totales", i, get_v("Activo total", i) - get_v("Pasivo Total", i))
                
                if get_v("Inmobilizado material bruto", i) == 0.0:
                    set_v("Inmobilizado material bruto", i, get_v("Inmobilizado material neto", i) + abs(get_v("Depreciacion acumulada", i)))
                if get_v("Inmobilizado material neto", i) == 0.0:
                    set_v("Inmobilizado material neto", i, get_v("Inmobilizado material bruto", i) - abs(get_v("Depreciacion acumulada", i)))
                
                # SÍNTESIS DE DEUDA REFORZADA PARA ASEGURADORAS
                if get_v("Deuda total", i) == 0.0:
                    set_v("Deuda total", i, get_v("Prestamos de corto plazo", i) + get_v("Porcion corriente de la deuda a largo plazo", i) + get_v("Deuda a largo plazo", i))
                if get_v("Deuda a largo plazo", i) == 0.0 and get_v("Deuda total", i) != 0.0:
                    set_v("Deuda a largo plazo", i, get_v("Deuda total", i) - get_v("Prestamos de corto plazo", i) - get_v("Porcion corriente de la deuda a largo plazo", i))

                if get_v("EBITDA", i) == 0.0:
                    ebitda = get_v("Beneficio neto de la empresa", i) + abs(get_v("Gastos de impuestos", i)) + abs(get_v("Gastos por intereses", i)) + get_v("Depreciacion y amortizacion", i)
                    if ebitda != 0.0: set_v("EBITDA", i, ebitda)
                if get_v("Flujo de caja libre", i) == 0.0:
                    set_v("Flujo de caja libre", i, get_v("Efectivo de Operaciones", i) - abs(get_v("Gastos de capital", i)))

        # --- DETECCIÓN DE STOCK SPLITS RETROACTIVOS ---
        shares_series = aligned_series.get("Promedio ponderado de acciones diluidas en circulacion", [0.0]*len(sorted_years))
        split_multipliers = [1.0] * len(sorted_years)
        cumulative_split = 1.0

        for i in range(len(sorted_years) - 1, 0, -1):
            shares_t = shares_series[i]
            shares_prev = shares_series[i-1]
            if shares_t > 0 and shares_prev > 0:
                ratio = shares_t / shares_prev
                matched_split = None
                if ratio > 1.2 or ratio < 0.8:
                    for std in [1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 50, 100]:
                        if abs(ratio - std) / std < 0.15:
                            matched_split = std
                            break
                    if not matched_split:
                        for std in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 50, 100]:
                            rev = 1.0 / std
                            if abs(ratio - rev) / rev < 0.15:
                                matched_split = rev
                                break
                if matched_split:
                    cumulative_split *= matched_split
                    logger.info(f"[{self.agent_name}] STOCK SPLIT DETECTADO: Ajustando año {sorted_years[i-1]}")
            split_multipliers[i-1] = cumulative_split

        for i in range(len(sorted_years)):
            mult = split_multipliers[i]
            if mult != 1.0:
                for metric in ["Promedio ponderado de acciones diluidas en circulacion", "Promedio ponderado de acciones basicas en circulacion", "Total de acciones fuera. en la fecha de presentacion"]:
                    if aligned_series.get(metric) and aligned_series[metric][i] != 0: aligned_series[metric][i] *= mult
                for metric in ["BPA diluido sin extraordinarios", "BPA basico", "Dividendo por accion", "Dividendo especial por accion", "Valor contable / Accion", "Flujo de caja por accion"]:
                    if aligned_series.get(metric) and aligned_series[metric][i] != 0: aligned_series[metric][i] /= mult

        normalized_data["aligned_series"] = aligned_series
        normalized_data["available_years_count"] = len(sorted_years)
        return normalized_data

    def _extract_fact_series(self, facts: Dict[str, Any], concept_names: List[str]) -> tuple:
        namespaces = list(facts.keys())
        def ns_priority(ns):
            if ns == 'us-gaap': return 0
            if ns == 'dei': return 1
            if ns == 'ifrs-full': return 2
            return 3
        namespaces.sort(key=ns_priority)

        merged_by_fy = {}

        for name in concept_names:
            for ns in namespaces:
                ns_facts = facts.get(ns, {})
                if name not in ns_facts: continue
                    
                units = ns_facts[name].get("units", {})
                key = None
                if "shares" in units and "Shares" in name: key = "shares"
                elif "USD/shares" in units: key = "USD/shares"
                elif "USD" in units: key = "USD"
                elif "pure" in units: key = "pure"
                elif "shares" in units: key = "shares"
                elif len(units) > 0: key = list(units.keys())[0]

                if not key: continue
                    
                items = units[key]
                sorted_items = sorted(items, key=lambda x: x.get("filed", "1970-01-01"))
                
                temp_fy_for_this_tag = {}
                for item in sorted_items:
                    if "val" not in item or "fy" not in item: continue
                        
                    form = item.get("form", "")
                    fp = item.get("fp", "")
                    fy_str = str(item["fy"])
                    
                    if form in ["10-K", "10-K/A", "20-F"] or fp == "FY":
                        if "start" in item and "end" in item:
                            try:
                                d_start = datetime.strptime(item["start"], "%Y-%m-%d")
                                d_end = datetime.strptime(item["end"], "%Y-%m-%d")
                                days = (d_end - d_start).days
                                if 330 <= days <= 380:
                                    temp_fy_for_this_tag[fy_str] = float(item["val"])
                            except: pass
                        else:
                            temp_fy_for_this_tag[fy_str] = float(item["val"])
                
                for y, val in temp_fy_for_this_tag.items():
                    if y not in merged_by_fy:
                        merged_by_fy[y] = val

        return merged_by_fy, list(merged_by_fy.keys())