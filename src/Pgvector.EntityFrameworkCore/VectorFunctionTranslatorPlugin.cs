﻿using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Diagnostics;
using Microsoft.EntityFrameworkCore.Query;
using Microsoft.EntityFrameworkCore.Query.SqlExpressions;
using Microsoft.EntityFrameworkCore.Storage;
using Npgsql.EntityFrameworkCore.PostgreSQL.Query.Expressions.Internal;
using System.Reflection;

namespace Pgvector.EntityFrameworkCore;

public class VectorFunctionTranslatorPlugin : IMethodCallTranslatorPlugin
{
    public VectorFunctionTranslatorPlugin(
        ISqlExpressionFactory sqlExpressionFactory, 
        IRelationalTypeMappingSource typeMappingSource
    )
    {
        Translators = new[]
        {
            new VectorFunctionTranslator(sqlExpressionFactory, typeMappingSource),
        };
    }

    public virtual IEnumerable<IMethodCallTranslator> Translators { get; }

    private class VectorFunctionTranslator : IMethodCallTranslator
    {
        private readonly ISqlExpressionFactory _sqlExpressionFactory;
        private readonly IRelationalTypeMappingSource _typeMappingSource;

        private static readonly MethodInfo _methodL2Distance = typeof(VectorExtensions)
            .GetRuntimeMethod(nameof(VectorExtensions.L2Distance), new[]
            {
                typeof(Vector),
                typeof(Vector),
            })!;

        private static readonly MethodInfo _methodCosineDistance = typeof(VectorExtensions)
            .GetRuntimeMethod(nameof(VectorExtensions.CosineDistance), new[]
            {
                typeof(Vector),
                typeof(Vector),
            })!;

        private static readonly MethodInfo _methodMaxInnerProduct = typeof(VectorExtensions)
            .GetRuntimeMethod(nameof(VectorExtensions.MaxInnerProduct), new[]
            {
                typeof(Vector),
                typeof(Vector),
            })!;

        public VectorFunctionTranslator(
            ISqlExpressionFactory sqlExpressionFactory, 
            IRelationalTypeMappingSource typeMappingSource
        )
        {
            _sqlExpressionFactory = sqlExpressionFactory;
            _typeMappingSource = typeMappingSource;
        }

#pragma warning disable EF1001
        public SqlExpression? Translate(
            SqlExpression? instance,
            MethodInfo method,
            IReadOnlyList<SqlExpression> arguments,
            IDiagnosticsLogger<DbLoggerCategory.Query> logger
        )
        {
            var vectorOperator = method switch
            {
                _ when ReferenceEquals(method, _methodL2Distance) => "<->",
                _ when ReferenceEquals(method, _methodCosineDistance) => "<=>",
                _ when ReferenceEquals(method, _methodMaxInnerProduct) => "<#>",
                _ => null
            };

            if (vectorOperator != null)
            {
                var left = arguments[0];
                var right = arguments[1];

                var resultTypeMapping = _typeMappingSource.FindMapping(method.ReturnType)!;

                return new PostgresUnknownBinaryExpression(
                    left: _sqlExpressionFactory.ApplyDefaultTypeMapping(left),
                    right: _sqlExpressionFactory.ApplyDefaultTypeMapping(right),
                    binaryOperator: vectorOperator,
                    type: resultTypeMapping.ClrType,
                    typeMapping: resultTypeMapping
                );
            }

            return null;
        }
#pragma warning restore EF1001
    }
}