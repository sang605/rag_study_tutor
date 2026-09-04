# DBMS Revision Notes (sample)

Replace this file with your own notes. It is only here so the project works
the moment you install it.

## Keys

A **primary key** is an attribute, or a set of attributes, that uniquely identifies
each row in a table. It cannot be NULL and it cannot repeat. A table has exactly one
primary key.

A **candidate key** is any minimal set of attributes that could serve as the primary
key. The one you choose becomes the primary key; the rest are alternate keys.

A **foreign key** is an attribute in one table that refers to the primary key of
another table. It is what enforces referential integrity: you cannot insert a row
whose foreign key points at a parent row that does not exist.

## Normalization

Normalization is the process of organising the columns and tables of a relational
database to reduce data redundancy and avoid update, insert and delete anomalies.

First Normal Form (1NF): every attribute holds a single atomic value, with no
repeating groups or multi-valued fields.

Second Normal Form (2NF): the table is in 1NF and every non-key attribute depends on
the whole primary key, not just part of it. Partial dependency is removed.

Third Normal Form (3NF): the table is in 2NF and no non-key attribute depends on
another non-key attribute. Transitive dependency is removed.

Boyce-Codd Normal Form (BCNF): a stricter 3NF. For every functional dependency
X to Y, X must be a super key.

Denormalization is the deliberate opposite: adding redundancy back to make reads
faster, usually in reporting or analytics systems.

## Transactions and ACID

A transaction is a single logical unit of work made of one or more operations. It
either completes fully or has no effect at all.

Atomicity: all operations in the transaction succeed, or none of them do.
Consistency: a transaction moves the database from one valid state to another valid state.
Isolation: concurrent transactions do not see each other's partial work.
Durability: once a transaction commits, its changes survive a crash or power loss.

Concurrency problems that isolation prevents include the dirty read (reading
uncommitted data), the non-repeatable read (the same row changes between two reads in
one transaction), and the phantom read (new rows appear that match an earlier query).

## Indexing

An index is a separate data structure, usually a B+ tree, that lets the database find
rows without scanning the whole table. It makes SELECT faster but slows INSERT, UPDATE
and DELETE, because the index must be maintained too, and it takes extra disk space.

A clustered index determines the physical order of rows in the table, so there can be
only one. A non-clustered index stores the key and a pointer to the row, and a table
may have many of them.

## Joins

An INNER JOIN returns only the rows that match in both tables.
A LEFT OUTER JOIN returns every row from the left table, with NULLs where the right
table has no match. RIGHT OUTER JOIN is the mirror image.
A FULL OUTER JOIN returns unmatched rows from both sides.
A CROSS JOIN returns the Cartesian product of both tables.
