#!/usr/bin/perl
=pod
Copyright (c) 2009-2014, UT-Battelle, LLC
All rights reserved

[PsimagLite, Version 1.0.0]

*********************************************************
THE SOFTWARE IS SUPPLIED BY THE COPYRIGHT HOLDERS AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED.

Please see full open source license included in file LICENSE.
*********************************************************

=cut
use warnings;
use strict;

package NewMake;

sub main
{
	local *FH = shift;
	my ($flavor, $args, $drivers, $additionals) = @_;
	die "newMake: needs additionals as 3rd argument\n" if (!defined($additionals));

	my %a = %$additionals;
	my $additional = $a{"additional"};
	my $additional2 = $a{"additional2"};
	my $additional3 = $a{"additional3"};
	my $path = $a{"path"};
	my $code = $a{"code"};
	$additional = " " unless defined($additional);
	$additional2 = " " unless defined($additional2);
	$additional3 = "" unless defined($additional3);
	$path = " " unless defined($path);

	my $allExecutables = combineAllDrivers($drivers,"");
	my $allCpps = combineAllDrivers($drivers,".cpp");

print FH<<EOF;
# DO NOT EDIT!!! Changes will be lost. Modify Config.make instead
# This Makefile was written by $0
# $code

CPPFLAGS += -I$path../../PsimagLite -I$path../../PsimagLite/src -I${path}Engine
all: $allExecutables $additional3

EOF

foreach my $ptr (@$drivers) {
	my $refptr = ref($ptr);
	my $oldmode = ($refptr eq "");
	my $what = ($oldmode) ? $ptr : $ptr->{"name"};
	my $aux = ($oldmode) ? 0 : $ptr->{"aux"};
	$aux = 0 if (!defined($aux));
	my $dotos = ($oldmode) ? "$what.o" : $ptr->{"dotos"};
	$dotos = "$what.o" if (!defined($dotos));

	print FH<<EOF;
$what.o: $what.cpp  Makefile $additional ${path}Config.make
	\$(CXX) \$(CPPFLAGS) -c $what.cpp

EOF

	if (!$aux) {
		# FIXME: Support many libs separated by commas here
		my $libs = ($oldmode) ? "" : $ptr->{"libs"};
		my $libs1 = "";
		my $libs2 = "";
		if (defined($libs) and $libs ne "") {
			$libs1 = "lib$libs.a";
			$libs2 = "-l$libs";
		}

		print FH<<EOF;
$what: $dotos $libs1
	\$(CXX) -o  $what $dotos \$(LDFLAGS) $libs2 \$(CPPFLAGS)
	\$(STRIP_COMMAND) $what

EOF
	}
}

print FH<<EOF;

$path../../PsimagLite/lib/libpsimaglite.a:
	\$(MAKE) -f Makefile -C $path../../PsimagLite/lib/

Makefile.dep: $allCpps $additional
	\$(CXX) \$(CPPFLAGS) -MM $allCpps  > Makefile.dep

clean::
	rm -f core* $allExecutables *.o *.dep $additional2

include Makefile.dep
EOF
}

sub combineAllDrivers
{
	my ($drivers,$extension) = @_;
	my $buffer = "";
	foreach my $ptr (@$drivers) {
		my $refptr = ref($ptr);
		my $oldmode = ($refptr eq "");
		my $what = ($oldmode) ? $ptr : $ptr->{"name"};
		my $aux = ($oldmode) ? 0 : $ptr->{"aux"};
		defined($aux) or $aux = 0;
		next if ($aux and $extension eq "");
		my $tmp = $what.$extension." ";
		$buffer .= $tmp;
	}

	return $buffer;
}

sub backupMakefile
{
	my ($dir) = @_;
	$dir = "." unless defined($dir);
	system("cp $dir/Makefile $dir/Makefile.bak") if (-r "$dir/Makefile");
	print STDERR "$0: Backup of $dir/Makefile in $dir/Makefile.bak\n";
}

1;
