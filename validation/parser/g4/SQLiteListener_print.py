# Generated from /Users/bhso/Documents/NLP_benchmark_paper/parser/g4/SQLite.g4 by ANTLR 4.7.1
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .SQLiteParser import SQLiteParser
else:
    from SQLiteParser import SQLiteParser

# This class defines a complete listener for a parse tree produced by SQLiteParser.
class SQLiteListener(ParseTreeListener):

    # Enter a parse tree produced by SQLiteParser#parse.
    def enterParse(self, ctx:SQLiteParser.ParseContext):
        print('ENTER: Parse ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#parse.
    def exitParse(self, ctx:SQLiteParser.ParseContext):
        print('EXIT: Parse ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#error.
    def enterError(self, ctx:SQLiteParser.ErrorContext):
        print('ENTER: Error ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#error.
    def exitError(self, ctx:SQLiteParser.ErrorContext):
        print('EXIT: Error ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Enter a parse tree produced by SQLiteParser#select_or_values.
    def enterSelect_or_values(self, ctx:SQLiteParser.Select_or_valuesContext):
        print('ENTER: Select_or_values ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#select_or_values.
    def exitSelect_or_values(self, ctx:SQLiteParser.Select_or_valuesContext):
        print('EXIT: Select_or_values ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Enter a parse tree produced by SQLiteParser#column_def.
    def enterColumn_def(self, ctx:SQLiteParser.Column_defContext):
        print('ENTER: Column_def ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#column_def.
    def exitColumn_def(self, ctx:SQLiteParser.Column_defContext):
        print('EXIT: Column_def ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#type_name.
    def enterType_name(self, ctx:SQLiteParser.Type_nameContext):
        print('ENTER: Type_name ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#type_name.
    def exitType_name(self, ctx:SQLiteParser.Type_nameContext):
        print('EXIT: Type_name ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#column_constraint.
    def enterColumn_constraint(self, ctx:SQLiteParser.Column_constraintContext):
        print('ENTER: Column_constraint ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#column_constraint.
    def exitColumn_constraint(self, ctx:SQLiteParser.Column_constraintContext):
        print('EXIT: Column_constraint ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#conflict_clause.
    def enterConflict_clause(self, ctx:SQLiteParser.Conflict_clauseContext):
        print('ENTER: Conflict_clause ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#conflict_clause.
    def exitConflict_clause(self, ctx:SQLiteParser.Conflict_clauseContext):
        print('EXIT: Conflict_clause ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#expr.
    def enterExpr(self, ctx:SQLiteParser.ExprContext):
        print('ENTER: Expr ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#expr.
    def exitExpr(self, ctx:SQLiteParser.ExprContext):
        print('EXIT: Expr ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#foreign_key_clause.
    def enterForeign_key_clause(self, ctx:SQLiteParser.Foreign_key_clauseContext):
        print('ENTER: Foreign_key_clause ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#foreign_key_clause.
    def exitForeign_key_clause(self, ctx:SQLiteParser.Foreign_key_clauseContext):
        print('EXIT: Foreign_key_clause ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#raise_function.
    def enterRaise_function(self, ctx:SQLiteParser.Raise_functionContext):
        print('ENTER: Raise_function ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#raise_function.
    def exitRaise_function(self, ctx:SQLiteParser.Raise_functionContext):
        print('EXIT: Raise_function ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#indexed_column.
    def enterIndexed_column(self, ctx:SQLiteParser.Indexed_columnContext):
        print('ENTER: Indexed_column ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#indexed_column.
    def exitIndexed_column(self, ctx:SQLiteParser.Indexed_columnContext):
        print('EXIT: Indexed_column ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#table_constraint.
    def enterTable_constraint(self, ctx:SQLiteParser.Table_constraintContext):
        print('ENTER: Table_constraint ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#table_constraint.
    def exitTable_constraint(self, ctx:SQLiteParser.Table_constraintContext):
        print('EXIT: Table_constraint ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#with_clause.
    def enterWith_clause(self, ctx:SQLiteParser.With_clauseContext):
        print('ENTER: With_clause ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#with_clause.
    def exitWith_clause(self, ctx:SQLiteParser.With_clauseContext):
        print('EXIT: With_clause ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#qualified_table_name.
    def enterQualified_table_name(self, ctx:SQLiteParser.Qualified_table_nameContext):
        print('ENTER: Qualified_table_name ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#qualified_table_name.
    def exitQualified_table_name(self, ctx:SQLiteParser.Qualified_table_nameContext):
        print('EXIT: Qualified_table_name ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#ordering_term.
    def enterOrdering_term(self, ctx:SQLiteParser.Ordering_termContext):
        print('ENTER: Ordering_term ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#ordering_term.
    def exitOrdering_term(self, ctx:SQLiteParser.Ordering_termContext):
        print('EXIT: Ordering_term ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#pragma_value.
    def enterPragma_value(self, ctx:SQLiteParser.Pragma_valueContext):
        print('ENTER: Pragma_value ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#pragma_value.
    def exitPragma_value(self, ctx:SQLiteParser.Pragma_valueContext):
        print('EXIT: Pragma_value ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#common_table_expression.
    def enterCommon_table_expression(self, ctx:SQLiteParser.Common_table_expressionContext):
        print('ENTER: Common_table_expression ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#common_table_expression.
    def exitCommon_table_expression(self, ctx:SQLiteParser.Common_table_expressionContext):
        print('EXIT: Common_table_expression ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#result_column.
    def enterResult_column(self, ctx:SQLiteParser.Result_columnContext):
        print('ENTER: Result_column ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#result_column.
    def exitResult_column(self, ctx:SQLiteParser.Result_columnContext):
        print('EXIT: Result_column ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#table_or_subquery.
    def enterTable_or_subquery(self, ctx:SQLiteParser.Table_or_subqueryContext):
        print('ENTER: Table_or_subquery ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#table_or_subquery.
    def exitTable_or_subquery(self, ctx:SQLiteParser.Table_or_subqueryContext):
        print('EXIT: Table_or_subquery ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#join_clause.
    def enterJoin_clause(self, ctx:SQLiteParser.Join_clauseContext):
        print('ENTER: Join_clause ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#join_clause.
    def exitJoin_clause(self, ctx:SQLiteParser.Join_clauseContext):
        print('EXIT: Join_clause ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#join_operator.
    def enterJoin_operator(self, ctx:SQLiteParser.Join_operatorContext):
        print('ENTER: Join_operator ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#join_operator.
    def exitJoin_operator(self, ctx:SQLiteParser.Join_operatorContext):
        print('EXIT: Join_operator ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#join_constraint.
    def enterJoin_constraint(self, ctx:SQLiteParser.Join_constraintContext):
        print('ENTER: Join_constraint ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#join_constraint.
    def exitJoin_constraint(self, ctx:SQLiteParser.Join_constraintContext):
        print('EXIT: Join_constraint ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#select_core.
    def enterSelect_core(self, ctx:SQLiteParser.Select_coreContext):
        # print('ENTER: Select_core ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#select_core.
    def exitSelect_core(self, ctx:SQLiteParser.Select_coreContext):
        # print('EXIT: Select_core ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#compound_operator.
    def enterCompound_operator(self, ctx:SQLiteParser.Compound_operatorContext):
        print('ENTER: Compound_operator ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#compound_operator.
    def exitCompound_operator(self, ctx:SQLiteParser.Compound_operatorContext):
        print('EXIT: Compound_operator ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#cte_table_name.
    def enterCte_table_name(self, ctx:SQLiteParser.Cte_table_nameContext):
        print('ENTER: Cte_table_name ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#cte_table_name.
    def exitCte_table_name(self, ctx:SQLiteParser.Cte_table_nameContext):
        print('EXIT: Cte_table_name ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#signed_number.
    def enterSigned_number(self, ctx:SQLiteParser.Signed_numberContext):
        print('ENTER: Signed_number ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#signed_number.
    def exitSigned_number(self, ctx:SQLiteParser.Signed_numberContext):
        print('EXIT: Signed_number ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#literal_value.
    def enterLiteral_value(self, ctx:SQLiteParser.Literal_valueContext):
        print('ENTER: Literal_value ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#literal_value.
    def exitLiteral_value(self, ctx:SQLiteParser.Literal_valueContext):
        print('EXIT: Literal_value ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#unary_operator.
    def enterUnary_operator(self, ctx:SQLiteParser.Unary_operatorContext):
        print('ENTER: Unary_operator ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#unary_operator.
    def exitUnary_operator(self, ctx:SQLiteParser.Unary_operatorContext):
        print('EXIT: Unary_operator ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#error_message.
    def enterError_message(self, ctx:SQLiteParser.Error_messageContext):
        print('ENTER: Error_message ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#error_message.
    def exitError_message(self, ctx:SQLiteParser.Error_messageContext):
        print('EXIT: Error_message ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#module_argument.
    def enterModule_argument(self, ctx:SQLiteParser.Module_argumentContext):
        print('ENTER: Module_argument ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#module_argument.
    def exitModule_argument(self, ctx:SQLiteParser.Module_argumentContext):
        print('EXIT: Module_argument ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#column_alias.
    def enterColumn_alias(self, ctx:SQLiteParser.Column_aliasContext):
        print('ENTER: Column_alias ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#column_alias.
    def exitColumn_alias(self, ctx:SQLiteParser.Column_aliasContext):
        print('EXIT: Column_alias ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#select_clause.
    def enterSelect_clause(self, ctx:SQLiteParser.Select_clauseContext):
        print('ENTER: Select_clause ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#select_clause.
    def exitSelect_clause(self, ctx:SQLiteParser.Select_clauseContext):
        print('EXIT: Select_clause ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#from_clause.
    def enterFrom_clause(self, ctx:SQLiteParser.From_clauseContext):
        print('ENTER: From_clause ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#from_clause.
    def exitFrom_clause(self, ctx:SQLiteParser.From_clauseContext):
        print('EXIT: From_clause ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#having_clause.
    def enterHaving_clause(self, ctx:SQLiteParser.Having_clauseContext):
        print('ENTER: Having_clause ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#having_clause.
    def exitHaving_clause(self, ctx:SQLiteParser.Having_clauseContext):
        print('EXIT: Having_clause ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#group_clause.
    def enterGroup_clause(self, ctx:SQLiteParser.Group_clauseContext):
        print('ENTER: Group_clause ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#group_clause.
    def exitGroup_clause(self, ctx:SQLiteParser.Group_clauseContext):
        print('EXIT: Group_clause ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#where_clause.
    def enterWhere_clause(self, ctx:SQLiteParser.Where_clauseContext):
        print('ENTER: Where_clause ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#where_clause.
    def exitWhere_clause(self, ctx:SQLiteParser.Where_clauseContext):
        print('EXIT: Where_clause ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#order_clause.
    def enterOrder_clause(self, ctx:SQLiteParser.Order_clauseContext):
        print('ENTER: Order_clause ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#order_clause.
    def exitOrder_clause(self, ctx:SQLiteParser.Order_clauseContext):
        print('EXIT: Order_clause ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#limit_clause.
    def enterLimit_clause(self, ctx:SQLiteParser.Limit_clauseContext):
        print('ENTER: Limit_clause ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#limit_clause.
    def exitLimit_clause(self, ctx:SQLiteParser.Limit_clauseContext):
        print('EXIT: Limit_clause ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#like_clause.
    def enterLike_clause(self, ctx:SQLiteParser.Like_clauseContext):
        print('ENTER: Like_clause ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#like_clause.
    def exitLike_clause(self, ctx:SQLiteParser.Like_clauseContext):
        print('EXIT: Like_clause ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#case_clause.
    def enterCase_clause(self, ctx:SQLiteParser.Case_clauseContext):
        print('ENTER: Case_clause ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#case_clause.
    def exitCase_clause(self, ctx:SQLiteParser.Case_clauseContext):
        print('EXIT: Case_clause ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#binary_operator.
    def enterBinary_operator(self, ctx:SQLiteParser.Binary_operatorContext):
        print('ENTER: Binary_operator ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#binary_operator.
    def exitBinary_operator(self, ctx:SQLiteParser.Binary_operatorContext):
        print('EXIT: Binary_operator ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#k_distinct.
    def enterK_distinct(self, ctx:SQLiteParser.K_distinctContext):
        print('ENTER: K_distinct ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#k_distinct.
    def exitK_distinct(self, ctx:SQLiteParser.K_distinctContext):
        print('EXIT: K_distinct ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#k_all.
    def enterK_all(self, ctx:SQLiteParser.K_allContext):
        print('ENTER: K_all ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#k_all.
    def exitK_all(self, ctx:SQLiteParser.K_allContext):
        print('EXIT: K_all ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Enter a parse tree produced by SQLiteParser#k_and.
    def enterK_and(self, ctx:SQLiteParser.K_andContext):
        print('ENTER: K_and ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#k_and.
    def exitK_and(self, ctx:SQLiteParser.K_andContext):
        print('EXIT: K_and ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#k_or.
    def enterK_or(self, ctx:SQLiteParser.K_orContext):
        print('ENTER: K_or ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#k_or.
    def exitK_or(self, ctx:SQLiteParser.K_orContext):
        print('EXIT: K_or ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#k_asc.
    def enterK_asc(self, ctx:SQLiteParser.K_ascContext):
        print('ENTER: K_asc ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#k_asc.
    def exitK_asc(self, ctx:SQLiteParser.K_ascContext):
        print('EXIT: K_asc ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#k_between.
    def enterK_between(self, ctx:SQLiteParser.K_betweenContext):
        print('ENTER: K_between ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#k_between.
    def exitK_between(self, ctx:SQLiteParser.K_betweenContext):
        print('EXIT: K_between ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#k_natural.
    def enterK_natural(self, ctx:SQLiteParser.K_naturalContext):
        print('ENTER: K_natural ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#k_natural.
    def exitK_natural(self, ctx:SQLiteParser.K_naturalContext):
        print('EXIT: K_natural ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#k_left.
    def enterK_left(self, ctx:SQLiteParser.K_leftContext):
        print('ENTER: K_left ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#k_left.
    def exitK_left(self, ctx:SQLiteParser.K_leftContext):
        print('EXIT: K_left ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#k_outer.
    def enterK_outer(self, ctx:SQLiteParser.K_outerContext):
        print('ENTER: K_outer ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#k_outer.
    def exitK_outer(self, ctx:SQLiteParser.K_outerContext):
        print('EXIT: K_outer ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#k_inner.
    def enterK_inner(self, ctx:SQLiteParser.K_innerContext):
        print('ENTER: K_inner ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#k_inner.
    def exitK_inner(self, ctx:SQLiteParser.K_innerContext):
        print('EXIT: K_inner ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#k_cross.
    def enterK_cross(self, ctx:SQLiteParser.K_crossContext):
        print('ENTER: K_cross ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#k_cross.
    def exitK_cross(self, ctx:SQLiteParser.K_crossContext):
        print('EXIT: K_cross ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#k_join.
    def enterK_join(self, ctx:SQLiteParser.K_joinContext):
        print('ENTER: K_join ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#k_join.
    def exitK_join(self, ctx:SQLiteParser.K_joinContext):
        print('EXIT: K_join ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#k_current_date.
    def enterK_current_date(self, ctx:SQLiteParser.K_current_dateContext):
        print('ENTER: K_current_date ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#k_current_date.
    def exitK_current_date(self, ctx:SQLiteParser.K_current_dateContext):
        print('EXIT: K_current_date ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#k_current_time.
    def enterK_current_time(self, ctx:SQLiteParser.K_current_timeContext):
        print('ENTER: K_current_time ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#k_current_time.
    def exitK_current_time(self, ctx:SQLiteParser.K_current_timeContext):
        print('EXIT: K_current_time ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#k_current_timestamp.
    def enterK_current_timestamp(self, ctx:SQLiteParser.K_current_timestampContext):
        print('ENTER: K_current_timestamp ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#k_current_timestamp.
    def exitK_current_timestamp(self, ctx:SQLiteParser.K_current_timestampContext):
        print('EXIT: K_current_timestamp ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#k_null.
    def enterK_null(self, ctx:SQLiteParser.K_nullContext):
        print('ENTER: K_null ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#k_null.
    def exitK_null(self, ctx:SQLiteParser.K_nullContext):
        print('EXIT: K_null ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#numeric_literal.
    def enterNumeric_literal(self, ctx:SQLiteParser.Numeric_literalContext):
        print('ENTER: Numeric_literal ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#numeric_literal.
    def exitNumeric_literal(self, ctx:SQLiteParser.Numeric_literalContext):
        print('EXIT: Numeric_literal ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#string_literal.
    def enterString_literal(self, ctx:SQLiteParser.String_literalContext):
        print('ENTER: String_literal ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#string_literal.
    def exitString_literal(self, ctx:SQLiteParser.String_literalContext):
        print('EXIT: String_literal ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#blob_literal.
    def enterBlob_literal(self, ctx:SQLiteParser.Blob_literalContext):
        print('ENTER: Blob_literal ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#blob_literal.
    def exitBlob_literal(self, ctx:SQLiteParser.Blob_literalContext):
        print('EXIT: Blob_literal ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#k_desc.
    def enterK_desc(self, ctx:SQLiteParser.K_descContext):
        print('ENTER: K_desc ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#k_desc.
    def exitK_desc(self, ctx:SQLiteParser.K_descContext):
        print('EXIT: K_desc ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#k_is.
    def enterK_is(self, ctx:SQLiteParser.K_isContext):
        print('ENTER: K_is ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#k_is.
    def exitK_is(self, ctx:SQLiteParser.K_isContext):
        print('EXIT: K_is ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#k_in.
    def enterK_in(self, ctx:SQLiteParser.K_inContext):
        print('ENTER: K_in ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#k_in.
    def exitK_in(self, ctx:SQLiteParser.K_inContext):
        print('EXIT: K_in ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#k_not.
    def enterK_not(self, ctx:SQLiteParser.K_notContext):
        print('ENTER: K_not ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#k_not.
    def exitK_not(self, ctx:SQLiteParser.K_notContext):
        print('EXIT: K_not ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#k_exists.
    def enterK_exists(self, ctx:SQLiteParser.K_existsContext):
        print('ENTER: K_exists ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#k_exists.
    def exitK_exists(self, ctx:SQLiteParser.K_existsContext):
        print('EXIT: K_exists ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#keyword.
    def enterKeyword(self, ctx:SQLiteParser.KeywordContext):
        print('ENTER: Keyword ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#keyword.
    def exitKeyword(self, ctx:SQLiteParser.KeywordContext):
        print('EXIT: Keyword ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#name.
    def enterName(self, ctx:SQLiteParser.NameContext):
        print('ENTER: Name ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#name.
    def exitName(self, ctx:SQLiteParser.NameContext):
        print('EXIT: Name ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#function_name.
    def enterFunction_name(self, ctx:SQLiteParser.Function_nameContext):
        print('ENTER: Function_name ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#function_name.
    def exitFunction_name(self, ctx:SQLiteParser.Function_nameContext):
        print('EXIT: Function_name ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#database_name.
    def enterDatabase_name(self, ctx:SQLiteParser.Database_nameContext):
        print('ENTER: Database_name ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#database_name.
    def exitDatabase_name(self, ctx:SQLiteParser.Database_nameContext):
        print('EXIT: Database_name ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#table_name.
    def enterTable_name(self, ctx:SQLiteParser.Table_nameContext):
        print('ENTER: Table_name ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#table_name.
    def exitTable_name(self, ctx:SQLiteParser.Table_nameContext):
        print('EXIT: Table_name ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#table_or_index_name.
    def enterTable_or_index_name(self, ctx:SQLiteParser.Table_or_index_nameContext):
        print('ENTER: Table_or_index_name ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#table_or_index_name.
    def exitTable_or_index_name(self, ctx:SQLiteParser.Table_or_index_nameContext):
        print('EXIT: Table_or_index_name ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#new_table_name.
    def enterNew_table_name(self, ctx:SQLiteParser.New_table_nameContext):
        print('ENTER: New_table_name ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#new_table_name.
    def exitNew_table_name(self, ctx:SQLiteParser.New_table_nameContext):
        print('EXIT: New_table_name ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#column_name.
    def enterColumn_name(self, ctx:SQLiteParser.Column_nameContext):
        print('ENTER: Column_name ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#column_name.
    def exitColumn_name(self, ctx:SQLiteParser.Column_nameContext):
        print('EXIT: Column_name ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#collation_name.
    def enterCollation_name(self, ctx:SQLiteParser.Collation_nameContext):
        print('ENTER: Collation_name ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#collation_name.
    def exitCollation_name(self, ctx:SQLiteParser.Collation_nameContext):
        print('EXIT: Collation_name ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#foreign_table.
    def enterForeign_table(self, ctx:SQLiteParser.Foreign_tableContext):
        print('ENTER: Foreign_table ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#foreign_table.
    def exitForeign_table(self, ctx:SQLiteParser.Foreign_tableContext):
        print('EXIT: Foreign_table ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#index_name.
    def enterIndex_name(self, ctx:SQLiteParser.Index_nameContext):
        print('ENTER: Index_name ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#index_name.
    def exitIndex_name(self, ctx:SQLiteParser.Index_nameContext):
        print('EXIT: Index_name ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#trigger_name.
    def enterTrigger_name(self, ctx:SQLiteParser.Trigger_nameContext):
        print('ENTER: Trigger_name ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#trigger_name.
    def exitTrigger_name(self, ctx:SQLiteParser.Trigger_nameContext):
        print('EXIT: Trigger_name ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#view_name.
    def enterView_name(self, ctx:SQLiteParser.View_nameContext):
        print('ENTER: View_name ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#view_name.
    def exitView_name(self, ctx:SQLiteParser.View_nameContext):
        print('EXIT: View_name ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#module_name.
    def enterModule_name(self, ctx:SQLiteParser.Module_nameContext):
        print('ENTER: Module_name ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#module_name.
    def exitModule_name(self, ctx:SQLiteParser.Module_nameContext):
        print('EXIT: Module_name ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#pragma_name.
    def enterPragma_name(self, ctx:SQLiteParser.Pragma_nameContext):
        print('ENTER: Pragma_name ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#pragma_name.
    def exitPragma_name(self, ctx:SQLiteParser.Pragma_nameContext):
        print('EXIT: Pragma_name ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#savepoint_name.
    def enterSavepoint_name(self, ctx:SQLiteParser.Savepoint_nameContext):
        print('ENTER: Savepoint_name ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#savepoint_name.
    def exitSavepoint_name(self, ctx:SQLiteParser.Savepoint_nameContext):
        print('EXIT: Savepoint_name ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#table_alias.
    def enterTable_alias(self, ctx:SQLiteParser.Table_aliasContext):
        print('ENTER: Table_alias ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#table_alias.
    def exitTable_alias(self, ctx:SQLiteParser.Table_aliasContext):
        print('EXIT: Table_alias ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#transaction_name.
    def enterTransaction_name(self, ctx:SQLiteParser.Transaction_nameContext):
        print('ENTER: Transaction_name ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#transaction_name.
    def exitTransaction_name(self, ctx:SQLiteParser.Transaction_nameContext):
        print('EXIT: Transaction_name ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


    # Enter a parse tree produced by SQLiteParser#any_name.
    def enterAny_name(self, ctx:SQLiteParser.Any_nameContext):
        # print('ENTER: Any_name ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass

    # Exit a parse tree produced by SQLiteParser#any_name.
    def exitAny_name(self, ctx:SQLiteParser.Any_nameContext):
        # print('EXIT: Any_name ({}, {}, {})'.format(ctx.getText(), ctx.getChildCount(), ctx.depth()))
        pass


