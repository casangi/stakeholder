#! /usr/bin/python3
""" Synchronizes changes between .py UnitTest files and .ipynb Jupyter files (originally for VLASS 1.2 tests).

This script DOES NOT create new .py or new .ipynb files, and any sections added/removed should be manually managed.
"""

import os
from pathlib import Path
import re
import json

class Section():
	""" Code section (either .ipynb or .py) """
	start_pattern = re.compile(r"[ \t]*# %%[ \t]*(.*?)[ \t]*start.*@")
	end_pattern   = re.compile(r"[ \t]*# %%[ \t]*(.*?)[ \t]*end.*@")

	def __init__(self, raw_code, name, source_file, line_start, line_end):
		self.raw_code = raw_code
		self.name = name
		self.source_file = source_file
		self.line_start = line_start
		self.line_end = line_end
		self.sync_code, self.sync_line_start, self.sync_line_end, self.indent, self.start_code, self.end_code = Section._parse_raw_code(raw_code, source_file, line_start)

		if self.sync_line_start > line_end:
			raise RuntimeError(f"Programmer error: failed to identify start of sync code in section {name} at {source_file}:{line_start}")
		if self.sync_line_end > line_end:
			raise RuntimeError(f"Programmer error: failed to identify end of sync code in section {name} at {source_file}:{line_start}")

	def _parse_raw_code(raw_code, source_file, line_start):
		""" Ignore any comments followed by any number of whitespace lines at the beggining and end.
		There MUST be a whitespace line after any comments at the beggining and beforey any comments at the end.
		
		Example format:
		# section start @

		pass

		# section end @
		"""
		if type(raw_code) == list: # as from a notebook
			raw_code = "".join(raw_code)
		lines = split_lines(raw_code)

		def unindent(txt, indent):
			lines = split_lines(txt)
			lines = [line.replace(indent, "", 1) for line in lines]
			return "\n".join(lines)

		i = -1 # current line index
		indent = -1
		found_start = False
		found_code = False
		found_end = False
		sync_code = ""
		sync_line_start = -1
		sync_line_end = -1
		start_lines = ""
		end_lines = ""
		for line in lines:
			i += 1
			line_s = line.lstrip()
			if not found_start:
				if line_s != "" and line_s[0] == '#':
					start_lines += line + "\n"
					continue
				elif line_s == "":
					start_lines += line + "\n"
					found_start = True
				else:
					raise RuntimeError(f"Expected a whitespace line in section at {source_file}:{line_start} before code started at line {line_start+i}")
			else: # if found_start
				if line_s == "":
					# whitespace line
					if not found_code:
						start_lines += line + "\n"
						continue # ignore any number of whitespace lines at the beggining
					end_lines += line + "\n"
					if not found_end:
						sync_line_end = line_start + i - 1
						found_end = True # could possibly be the ignored whitespace lines before the end
				elif line_s[0] == '#' and found_end:
					# comment line, possibly at the end of the section
					end_lines += line + "\n"
				else:
					# code line
					found_end = False # need to find whitespace before ending
					if end_lines != "": # include any skipped whitespace and comment lines
						sync_code += unindent(end_lines, indent)
						end_lines = ""
					if not found_code:
						found_code = True
						sync_line_start = line_start + i
						indent = line.replace(line_s, "")
					sync_code += unindent(line, indent) + "\n"

		if not found_start:
			raise RuntimeError(f"Didn't find any code in section at {source_file}:{line_start}")
		if not found_end:
			raise RuntimeError(f"Didn't find the end of the section at {source_file}:{line_start}. Make sure there's an empty line before the section end.")
		if sync_code == "" or not found_code:
			raise RuntimeError(f"Didn't find any code in the section at {source_file}:{line_start}")

		return sync_code, sync_line_start, sync_line_end, indent, start_lines, end_lines

class NotebookSection(Section):
	def set_nb_vals(self, cell_idx):
		self.cell_idx = cell_idx

def split_lines(txt):
	return txt.replace("\r\n", "\n").replace("\n\r", "\n").split("\n")

def find_files():
	currdir = Path(os.getcwd())
	basedir = Path(currdir).parent.absolute()

	# find the only recognized unittest .py file
	ut_name = str(Path(basedir, 'standard', 'test_vlass_1v2.py'))
	if not os.path.exists(ut_name):
		raise RuntimeError(f"Can't find unittest file {ut_name}")
	ut_files = [ut_name]

	# find jupyter notebook .ipynb files
	nb_names = list(currdir.glob("*.ipynb"))
	if len(nb_names) == 0:
		raise RuntimeError(f"Can't find any Jupyter notebook files at {currdir}")
	nb_files = [str(Path(currdir, nb_name)) for nb_name in nb_names]

	return ut_files, nb_files

def get_sections(file_name, file_or_lines, section_class=None):
	ret = []
	section_class = section_class if (section_class != None) else Section

	i = 0 # current line number
	in_section = False
	section_name = ""
	line_start = -1
	line_end = -1
	raw_code = ""
	for line in file_or_lines:
		i += 1
		start_match = Section.start_pattern.match(line)
		end_match = Section.end_pattern.match(line)

		if start_match != None:
			if not in_section:
				in_section = True
				section_name = start_match[1]
				line_start = i
			else:
				raise RuntimeError(f"Found unexpected section start in section \"{section_name}\" at {file_name}:{i} (previous section starts at line {line_start})")

		if in_section:
			raw_code += line

		if end_match != None:
			if not in_section:
				raise RuntimeError(f"Found unmatched end section at {file_name}:{i}")
			else: # if in_section
				end_name = end_match[1]
				if end_name != section_name:
					raise RuntimeError(f"Found unexpected end section \"{end_name}\" while in section \"{section_name}\" at {file_name}:{i} (section starts at line {line_start})")
				line_end = i			
				ret.append(section_class(raw_code, section_name, file_name, line_start, line_end))
				in_section = False
				section_name = ""
				line_start = -1
				line_end = -1
				raw_code = ""
	if in_section:
		raise RuntimeError(f"Found unmatched start section \"{section_name}\" at {file_name}:{line_start}")

	return ret

def get_ut_sections(ut_files):
	for ut_name in ut_files:
		with open(ut_name, 'r') as fin:
			sections = get_sections(ut_name, fin)
	ret = {}
	for section in sections:
		if section.name in ret:
			raise RuntimeError(f"Found duplicate section \"{section.name}\"")
		ret[section.name] = section
	return ret

def get_nb_sections(nb_files):
	sections = []
	for nb_name in nb_files:
		with open(nb_name, 'r') as fin:
			parsed = json.load(fin)
		for cell_idx in range(len(parsed['cells'])):
			cell = parsed['cells'][cell_idx]
			if cell['cell_type'] != 'code':
				continue
			source = cell['source']
			if len(source) > 1:
				source[-1] = source[-1].rstrip()
			cell_sections = get_sections(nb_name, source, section_class=NotebookSection)
			for section in cell_sections:
				section.set_nb_vals(cell_idx=cell_idx)
			sections += cell_sections
	ret = {}
	for section in sections:
		if section.name in ret:
			raise RuntimeError(f"Found duplicate section \"{section.name}\"")
		ret[section.name] = section
	return ret

def sync_section(ut_section, nb_section, mode='tonb'):
	if mode == 'tonb':
		nb_section.push_code = ut_section.sync_code
	elif mode == 'tout':
		ut_section.push_code = nb_section.sync_code
	return True

def update_sections_in_file(file_name, sections_to_update, mode='tonb'):
	def get_new_section_code(section):
		push_lines = split_lines(section.push_code)
		body_code = section.indent + ("\n"+section.indent).join(push_lines)
		return section.start_code + body_code + section.end_code

	if mode == 'tonb':
		with open(file_name, 'r') as fin:
			parsed = json.load(fin)
		for section in sections_to_update:
			new_source_lines = split_lines(get_new_section_code(section))
			new_source_lines = new_source_lines[:-1] # split_lines adds an extra line at the end
			new_source_lines = [line+'\n' for line in new_source_lines] # .ipynb json needs extra '\n' characters
			parsed['cells'][section.cell_idx]['source'] = new_source_lines
		outstr = json.dumps(parsed, indent=2)
		with open(file_name, 'w') as fout:
			fout.write(outstr)
	else:
		with open(file_name, 'r') as fin:
			lines = fin.readlines()
		sections_to_update = sections_to_update.sort(key=lambda s: s.line_start, reverse=True)
		for section in sections_to_update:
			before_lines = [] if (section.line_start == 0)       else lines[:section.line_start-1]
			after_lines  = [] if (section.line_end < len(lines)) else lines[section.line_end+1:]
			lines = before_lines + [get_new_section_code(section)] + after_lines
		with open(file_name, 'w') as fout:
			fout.writelines(lines)

if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description='Synchronizes .py UnitTest and .ipynb Jupyter files (originally for VLASS 1.2 tests)')
	group = parser.add_mutually_exclusive_group(required=True)
	group.add_argument('--tonb', action='store_const', const='tonb', dest='mode', help='to NoteBook (from .py to .ipynb)')
	group.add_argument('--tout', action='store_const', const='tout', dest='mode', help='to UnitTest (from .ipynb to .py)')
	parser.add_argument('--dryrun', action='store_true', help='Dry run, don\'t modify any files')
	parser.add_argument('--verbose', '-v', action='count', default=0)

	args = parser.parse_args()
	
	# catalog all sections
	ut_files, nb_files = find_files()
	ut_sections = get_ut_sections(ut_files)
	nb_sections = get_nb_sections(nb_files)
	if (args.verbose >= 2):
		for sections in [nb_sections]:#[ut_sections, nb_sections]:
			for section in sections.values():
				print(f"Section \"{section.name}\" [{section.source_file}]:")
				print(f"Raw code range: {section.line_start}-{section.line_end}")
				print(f"Sync code range: {section.sync_line_start}-{section.sync_line_end}")
				print(f"Indent: \"{section.indent}\"")
				print(f">>>\n{section.sync_code}<<<")
				print("")
	for k in ut_sections.keys():
		if k not in nb_sections:
			print(f"Warning: UnitTest section \"{k}\" was not found in any notebooks")
	for k in nb_sections.keys():
		if k not in ut_sections:
			print(f"Warning: Notebook section \"{k}\" was not found in any unittests")

	# find the differing sections
	differing_section_names = []
	differing_sections_by_files = {}
	for k in ut_sections.keys():
		if k not in nb_sections:
			continue
		ut_section = ut_sections[k]
		nb_section = nb_sections[k]
		if ut_section.sync_code != nb_section.sync_code:
			differing_section_names.append(k)
			to_section = nb_section if (args.mode == 'tonb') else ut_section
			file_name = to_section.source_file
			if file_name not in differing_sections_by_files:
				differing_sections_by_files[file_name] = []
			differing_sections_by_files[file_name].append(to_section)

	# update the sections
	for k in differing_section_names:
		if (args.verbose >= 1):
			arrow = '-->' if (args.mode == 'tonb') else '<--'
			print(f"{ut_sections[k].source_file} {arrow} {nb_sections[k].source_file} [{nb_sections[k].name}]")
		sync_section(ut_sections[k], nb_sections[k], args.mode)

	# save the sections back out
	if not args.dryrun:
		for file_name in differing_sections_by_files.keys():
			update_sections_in_file(file_name, differing_sections_by_files[file_name], args.mode)