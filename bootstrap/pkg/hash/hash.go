/*
 Serene --- Yet an other Lisp

Copyright (c) 2020  Sameer Rahmani <lxsameer@gnu.org>

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 2 of the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

// Package hash provides the hashing functionality
package hash

import "hash/crc32"

var hashTable *crc32.Table = crc32.MakeTable(crc32.Castagnoli)

//IHashable is the interface types which allows expressions to have a hash
// value that doesn't change through out their life time. The origin
// of each expression can be checked by comparing their hashes. Basically
// two expressions with the same hash consider to be the same.
type IHashable interface {
	// Returns a 32 bit hash of the the entity which implements it.
	// The hash should be constant to the life time of the implementor.
	Hash() uint32
}

func HashOf(in []byte) uint32 {
	return crc32.Checksum(in, hashTable)
}
