#include <iostream>
#include <fstream>
#include <elfio/elfio.hpp>
#include <elfio/elfio_dump.hpp>

using namespace ELFIO;

section *find_section( const elfio &elf, std::string section_name )
{
	Elf_Half sec_num = elf.sections.size();
	for ( Elf_Half i = 0; i < sec_num; i++ ) {
		section *psec = elf.sections[i];
		const std::string name = psec->get_name();
		if ( name.size() != 0 && name == section_name ) {
			return psec;
		}
	}
	dump::section_headers( std::cout, elf );
	return NULL;
}

template <typename T>
void modify_symbol_size( elfio &elf, section *psec, 
						 std::string kernel_name, Elf_Xword size)
{
	symbol_section_accessor symacc(elf, psec);

	for ( Elf_Xword index = 0; index < symacc.get_symbols_num(); index++ ) {
		char *pdata = new char[psec->get_size()];
		std::copy( psec->get_data(), psec->get_data() + psec->get_size(), pdata );
	    T* pSym = reinterpret_cast<T*>( pdata + index * psec->get_entry_size() );
	    const endianess_convertor& convertor = elf.get_convertor();
	    section* string_section = elf.sections[ ( Elf_Half )psec->get_link() ];
	    string_section_accessor str_reader( string_section );
	    const char *cname = str_reader.get_string( convertor( pSym->st_name ) );
	    if ( cname ) {
		    std::string name = cname;
		    std::cout << name << std::endl;
		    if ( name == kernel_name ) {
		    	pSym->st_size = convertor( size );
		    	psec->set_data( pdata, psec->get_size() );
		    }
		}
	    delete[] pdata;
	}

}

void dump_elf( std::ostream& out, elfio &elf )
{
	dump::header         ( out, elf );
    dump::section_headers( out, elf );
    dump::segment_headers( out, elf );
    dump::symbol_tables  ( out, elf );
    dump::notes          ( out, elf );
    dump::dynamic_tags   ( out, elf );
    dump::section_datas  ( out, elf );
    dump::segment_datas  ( out, elf );
}

int main( int argc, char *argv[] )
{
	if ( argc != 6 ) {
		std::cout << "Usage: g2d from_cubin to_cubin from_kernel to_kernel new_to" << std::endl;
		exit(0);
	}
	
	elfio from;
	if ( !from.load( argv[1] ) ) {
		std::cout << "Can't find or process ELF file " << argv[1] << std::endl;
		exit(0);
	}
	dump::section_headers( std::cout, from );
	dump::symbol_tables( std::cout, from );
	// std::ofstream ffrom("from_dump.txt");
	// if ( !ffrom.is_open() ) {
	// 	std::cout << "Failed open ffrom" << std::endl;
	// 	exit(0);
	// }
	// dump_elf( ffrom, from );
	// ffrom.close();

	elfio to;
	if ( !to.load( argv[2] ) ) {
		std::cout << "Can't find or process " << argv[2] << std::endl;
		exit(0);
	}
	dump::section_headers( std::cout, to );
	dump::symbol_tables( std::cout, to );
	// std::ofstream fto("to_dump.txt");
	// if ( !fto.is_open() ) {
	// 	std::cout << "Failed open fto" << std::endl;
	// 	exit(0);
	// }
	// dump_elf( fto, to );
	// fto.close();

	std::string from_kernel( argv[3] );
	std::string to_kernel( argv[4] );

	section *from_section = find_section( from, ".text." + from_kernel );
	if ( !from_section ) {
		std::cout << "Can't find " << argv[3] << " in " << argv[1] << std::endl;
		exit(0);
	}

	section *to_section = find_section( to, ".text." + to_kernel );
	if ( !to_section ) {
		std::cout << "Can't find " << argv[3] << " in " << argv[2] << std::endl;
		exit(0);
	}

	const char *from_data = from_section->get_data();
	Elf_Xword from_data_size = from_section->get_size();
	to_section->set_data( from_data, from_data_size );
	
	Elf_Word from_info = from_section->get_info();
	Elf_Word to_info = to_section->get_info();
	Elf_Word reg_mask = 0xff;
	to_section->set_info(
		( to_info & ( ~ ( reg_mask << 24 ) ) ) | ( from_info & ( reg_mask << 24 ) )
	);
	
	Elf_Xword from_flag = from_section->get_flags();
	Elf_Xword to_flag = to_section->get_flags();
	Elf_Xword bar_mask = 0x1f;
	to_section->set_flags(
		( to_flag & ( ~ ( bar_mask << 20 ) ) ) | ( from_flag & ( bar_mask << 20 ) )
	);

	section *symtab = find_section( to, ".symtab");
	if ( !symtab ) {
		std::cout << "Can't find .symtab" << std::endl;
		exit(0);
	}
    if ( to.get_class() == ELFCLASS32 ) {
        modify_symbol_size<Elf32_Sym>( to, symtab, to_kernel, to_section->get_size() );
    }
    else {
        modify_symbol_size<Elf64_Sym>( to, symtab, to_kernel, to_section->get_size() );
    }
	
	dump::section_headers( std::cout, to );
	dump::symbol_tables( std::cout, to );

    to.save( argv[5] );

	return 0;
}