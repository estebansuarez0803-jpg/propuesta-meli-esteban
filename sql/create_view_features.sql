CREATE TABLE IF NOT EXISTS clientes (
  cantidad_usuarios INT,
  edad INT,
  genero TEXT,
  marca_preferida TEXT,
  total_compras INT,
  frecuencia_de_compra INT,
  promociones_utilizadas INT
);

CREATE OR REPLACE VIEW vw_clientes_features AS
SELECT
  cantidad_usuarios,
  edad,
  genero,
  marca_preferida,
  total_compras,
  frecuencia_de_compra,
  promociones_utilizadas
FROM clientes;
