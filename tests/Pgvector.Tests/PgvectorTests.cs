using Xunit;
using Npgsql;
using Pgvector;

namespace Pgvector.Tests;

public class Example
{
    [Fact]
    public async Task Main()
    {
        var connString = "Host=localhost;Database=pgvector_dotnet_test";

        await using var conn = new NpgsqlConnection(connString);
        await conn.OpenAsync();

        await using (var cmd = new NpgsqlCommand("CREATE EXTENSION IF NOT EXISTS vector", conn))
        {
            await cmd.ExecuteNonQueryAsync();
        }

        await using (var cmd = new NpgsqlCommand("DROP TABLE IF EXISTS items", conn))
        {
            await cmd.ExecuteNonQueryAsync();
        }

        await using (var cmd = new NpgsqlCommand("CREATE TABLE items (embedding vector(3))", conn))
        {
            await cmd.ExecuteNonQueryAsync();
        }

        await using (var cmd = new NpgsqlCommand("INSERT INTO items (embedding) VALUES ($1::vector), ($2::vector), ($3::vector)", conn))
        {
            var embedding1 = new Vector(new float[] {1, 1, 1});
            var embedding2 = new Vector(new float[] {2, 2, 2});
            var embedding3 = new Vector(new float[] {1, 1, 2});
            cmd.Parameters.AddWithValue(embedding1.ToString());
            cmd.Parameters.AddWithValue(embedding2.ToString());
            cmd.Parameters.AddWithValue(embedding3.ToString());
            await cmd.ExecuteNonQueryAsync();
        }

        await using (var cmd = new NpgsqlCommand("SELECT * FROM items ORDER BY embedding <-> $1::vector LIMIT 5", conn))
        {
            var embedding = new Vector(new float[] {1, 1, 1});
            cmd.Parameters.AddWithValue(embedding.ToString());
            cmd.AllResultTypesAreUnknown = true;

            await using (var reader = await cmd.ExecuteReaderAsync())
            {
                while (await reader.ReadAsync())
                    Console.WriteLine((new Vector(reader.GetString(0))).ToString());
            }
        }

        await using (var cmd = new NpgsqlCommand("CREATE INDEX my_index ON items USING ivfflat (embedding vector_l2_ops)", conn))
        {
            await cmd.ExecuteNonQueryAsync();
        }
    }
}